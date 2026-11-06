"""One persistent local Python process: cached vision, command queue and STOP."""
from collections import deque
import json
from pathlib import Path
import queue
import threading
import time
import traceback
import cv2
import numpy as np
from .cameras import Cameras
from .geometry import Geometry
from .scene import Scene
from .robot import Robot
from .motion import Motion
from .actions import Actions
from .recovery import Recovery
from . import tasks


class Runtime:
    from .experiments import MODES as EXTRA_MODES
    MODES = ('relocate','stow','extract','collect_to_box','empty_box','sort_colors')+EXTRA_MODES

    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root/'wristlab/config.yaml').read_text())
        self.output = self.root/'runtime'
        self.output.mkdir(exist_ok=True)
        self.geometry = Geometry(self.config['native_root'])
        self.cameras = Cameras(self.config['api'],self.config['frame_max_age_s'],self.config.get('camera_controls'))
        self.robot = Robot(self.config['native_root'],self.output/'controller')
        self.scene = Scene(self.geometry,self.robot)
        self.stop_event,self.pause_event = threading.Event(),threading.Event()
        self.commands = queue.Queue(maxsize=1)
        self.submit_lock = threading.Lock()
        self.latch_lock = threading.RLock()
        self.stop_generation = 0
        self.stop_file = self.output/'stop-latch.json'
        previous = {}
        try:
            previous = json.loads(self.stop_file.read_text())
        except (OSError,ValueError):
            try:
                previous = json.loads((self.output/'status.json').read_text())
            except (OSError,ValueError):
                pass
        self.fault_latched = bool(previous.get('hardware_fault'))
        if previous.get('stop_latched'):
            self.stop_event.set()
        self.motion = Motion(self.robot,self.check_cancel,self.report,self.root/self.config['qualification_file'])
        self.recovery = Recovery(self.config['max_recovery_attempts'])
        self.actions = Actions(self)
        self.running,self.initialized = False,False
        self.phase,self.detail = 'initializing','Загрузка калибровки и детекторов'
        self.mode = None
        self.metrics,self.results = {},[]
        self.log = deque(maxlen=160)
        self.frame_history = deque(maxlen=24)
        self.last_plan = None
        self.physical_relocate_count = 0
        self.selected,self.destination = None,None
        self.last_refresh = self.started = self.finished = None
        self.last_native_state = None
        self.native_stop_ack = None
        self.native_stop_seen = None
        self.camera_error = None
        self.hardware_error = None
        self.report('initializing','No LLM, SAM or subprocess in experiment execution')

    def report(self, phase, detail):
        self.phase,self.detail = phase,str(detail)
        self.log.append(time.strftime('%H:%M:%S')+' '+phase+' '+str(detail))
        print(self.log[-1],flush=True)

    def initialize(self):
        cv2.setNumThreads(2)
        try:
            self.robot.connect_once(self.tick)
            self.motion.planner.command_sag = float(self.robot.arm.raw.sag)
            state = self.robot.read_state()
            self.cameras.record_state(state['timestamp'],state['q'],state['gripper'])
            audit = self.robot.audit_state(state)
            (self.output/'robot-audit.json').write_text(json.dumps(audit,indent=2))
        except Exception as exc:
            self.hardware_error = str(exc)
        try:
            self.metrics['camera_controls'] = self.cameras.apply_controls()
            self.cameras.start_capture()
            wrist_path = self.output/'wrist-camera-calibration.json'
            if wrist_path.exists():
                self.geometry.install_wrist_calibration(json.loads(wrist_path.read_text()),self.robot)
            self.cameras.on_native_stop = self.native_stop
            self.refresh()
        except Exception as exc:
            self.report('blocked',str(exc))
        self.initialized = True
        reasons = self.blockers()
        if not reasons:
            self.ensure_driver()
        self.report('blocked' if reasons else 'ready','; '.join(reasons) if reasons else 'Ready')
        self.persist()

    def refresh(self):
        begin = time.monotonic()
        try:
            frames = self.cameras.latest_pair()
            t = time.monotonic()
            from .vision import frame_quality
            qualities = {name:frame_quality(frame,40. if name=='overview' else 15.) for name,frame in frames.items()}
            self.metrics['camera_quality'] = qualities
            invalid = [name+': '+q['reason'] for name,q in qualities.items() if not q['usable']]
            if invalid:
                raise ValueError('; '.join(invalid))
            self.scene.update_scene(frames)
            self.metrics['vision_s'] = time.monotonic()-t
            self.metrics['camera_fetch_s'] = t-begin
            self.camera_error = None
            self.frame_history.append(dict(frames))
            self.last_refresh = time.time()
        except Exception as exc:
            self.camera_error = str(exc)
            raise

    def blockers(self):
        reasons = self.motion.qualification_reasons(self.geometry)
        if self.camera_error:
            reasons.append(self.camera_error)
        if self.last_refresh is None or time.time()-self.last_refresh > 2.:
            reasons.append('нет актуального наблюдения сцены')
        if not self.scene.state.boxes:
            reasons.append('проёмы коробок ещё не измерены')
        elif any(not b.evidence.get('metric_qualified') for b in self.scene.state.boxes.values()):
            reasons.append('геометрия дна/бортов коробок не квалифицирована')
        if self.stop_event.is_set():
            reasons.append('STOP зафиксирован')
        return reasons

    def ensure_driver(self):
        self.check_cancel()
        self.cameras.check()
        self.robot.connect_once(self.tick)
        state = self.robot.read_state()
        if not self.robot.audit_state(state)['valid']:
            self.robot.close()
            raise ValueError('measured current pose fails native constraints')
        self.cameras.record_state(state['timestamp'],state['q'],state['gripper'])
        self.motion.prepare_driver(self.cameras.check)

    def native_stop(self, token=None):
        with self.latch_lock:
            if token!=self.native_stop_seen:
                self.native_stop_seen = token
                self.stop_generation += 1
            if token!=self.native_stop_ack and not self.stop_event.is_set():
                self.stop(forward=False,reason='native STOP event '+str(token))

    def tick(self, record):
        self.check_cancel()
        self.cameras.check()
        self.motion.guarded_tick(record)
        stamp = time.time()
        self.cameras.record_state(stamp,self.robot.from_native(record['pos']),record['pos']['gripper'])
        if self.robot.first_command_at is None:
            self.robot.first_command_at = time.monotonic()

    def check_cancel(self):
        if self.stop_event.is_set():
            raise RuntimeError('STOP')
        if self.robot.arm is not None and not self.robot.owns_lease():
            raise RuntimeError('exclusive hardware lease lost')

    def wait_pause(self):
        while self.pause_event.is_set():
            self.check_cancel()
            self.cameras.check()
            if self.scene.held is not None:
                self.motion.driver.confirm_retention()
                state = self.robot.read_state()
                self.cameras.record_state(state['timestamp'],state['q'],state['gripper'])
            time.sleep(.05)
        self.check_cancel()

    def start(self, payload):
        mode = payload.get('mode','relocate')
        if mode not in self.MODES:
            raise ValueError('unsupported experiment')
        dry = bool(payload.get('dry',False))
        with self.submit_lock:
            if self.running or not self.commands.empty():
                raise ValueError('another task is already running')
            if self.stop_event.is_set():
                raise ValueError('STOP is latched; reset while idle before a new check')
            if not self.initialized:
                raise ValueError('runtime is initializing')
            if not dry and self.blockers():
                raise ValueError('; '.join(self.blockers()))
            self.robot.first_command_at = None
            self.started = time.monotonic()
            self.finished = None
            self.mode = mode
            self.scene.remember_reset_layout()
            self.running = True
            self.metrics = {'first_command_s':None,'cycle_s':None}
            self.results = []
            self.commands.put_nowait(payload | {'mode':mode,'dry':dry})
            self.report('queued',mode+(' · geometric check' if dry else ''))

    def stop(self, forward=True, reason='operator STOP', hardware_fault=False):
        with self.latch_lock:
            self.stop_event.set()
            self.stop_generation += 1
            self.fault_latched = self.fault_latched or hardware_fault
            try:
                self.stop_file.write_text(json.dumps({'stop_latched':True,'hardware_fault':self.fault_latched,
                                                     'reason':reason,'timestamp':time.time()},ensure_ascii=False))
            except OSError as exc:
                self.fault_latched = True
                self.log.append('STOP persistence failed: '+str(exc))
        self.pause_event.clear()
        self.robot.stop_motion()
        native = None
        if forward:
            try:
                native = self.cameras.json('/api/stop',{})
            except Exception as exc:
                native = {'error':str(exc)}
        self.report('stopped','STOP: native goals frozen; no automatic folding')
        return {'stopped':True,'native':native,'automatic_motion':False,'torque_changed':False}

    def reset(self):
        with self.submit_lock:
            if self.running or not self.commands.empty():
                raise ValueError('cannot reset during a running or queued task')
            self.commands.put_nowait({'control':'reset'})

    def perform_reset(self):
        generation = self.stop_generation
        native = self.cameras.json('/api/status')
        self.robot.close()
        self.robot.connect_once(self.tick)
        with self.latch_lock:
            if generation!=self.stop_generation:
                raise ValueError('new STOP arrived during reset; reset refused')
            self.native_stop_ack = native.get('updated_at') if native.get('state')=='EMERGENCY_STOP' else None
            self.stop_event.clear()
            self.fault_latched = False
            if self.stop_file.exists():
                self.stop_file.unlink()
            self.pause_event.clear()
        self.report('blocked','STOP сброшен вручную; чтение состояния, без запуска движения')

    def select(self, payload):
        camera_id,frame_id = payload['camera_id'],payload['frame_id']
        frames = next((frames for frames in reversed(list(self.frame_history)) if camera_id in frames and frames[camera_id].frame_id==frame_id),None)
        if frames is None or time.time()-frames[camera_id].timestamp > 3.:
            raise ValueError('clicked frame expired; click a current image')
        from .vision import detect_cubes, segment_clicked_object
        frame = frames[camera_id]
        detection = segment_clicked_object(frame,(payload['pixel_x'],payload['pixel_y']),detect_cubes(frame))
        objects = [o for o in self.scene.state.objects.values() if o.color==detection['color']]
        if len(objects) != 1:
            raise ValueError('camera association is ambiguous for multiple objects of this color')
        self.selected = objects[0].id
        return {'object_id':self.selected,'color':objects[0].color,'frame_id':frame_id}

    def execute(self, payload):
        try:
            self.check_cancel()
            if self.robot.arm is None:
                self.robot.read_state()
            self.refresh()
            selected = payload.get('object_id') or self.selected
            destination = payload.get('destination') or self.destination
            if payload['mode'] in self.EXTRA_MODES and not payload['dry']:
                from .experiments import execute
                self.results = execute(payload['mode'],self.scene,self.actions,self.refresh,self.check_cancel,**{k:v for k,v in payload.items() if k not in ('mode','dry')})
            elif payload['dry']:
                self.report('checking','contact-free preflight; actuator commands prohibited')
                reasons = self.blockers()
                if reasons:
                    raise ValueError('; '.join(reasons))
                if payload['mode'] in self.EXTRA_MODES:
                    from .experiments import prepare,next_step
                    options = {k:v for k,v in payload.items() if k not in ('mode','dry')}
                    if payload['mode']=='run_sequence':
                        raise ValueError('use task-plan for sequence; each physical stage requires its freshly observed scene')
                    item = next_step(self.scene,payload['mode'],prepare(self.scene,payload['mode'],options))
                else:
                    item = tasks.next_transfer(payload['mode'],self.scene,selected,destination)
                if item is not None:
                    self.results = [self.actions.relocate(*item,dry=True)]
            else:
                self.results = tasks.run_task(payload['mode'],self.scene,self.actions,self.refresh,self.check_cancel,selected,destination)
            ok = all(result.success for result in self.results)
            self.report('completed' if ok else 'blocked','Все запрошенные переносы подтверждены' if ok else self.results[-1].reason)
        except Exception as exc:
            self.report('blocked',f'{type(exc).__name__}: {exc}')
            self.results = [{'success':False,'reason':str(exc),'physical_verified':False}]
        finally:
            self.finished = time.monotonic()
            self.metrics['request_s'] = self.finished-self.started
            if self.robot.first_command_at is not None:
                self.metrics['first_command_s'] = self.robot.first_command_at-self.started
            self.running = False
            self.persist()

    def preview(self, payload):
        """Pure task geometry remains available while physical STOP is latched."""
        import copy
        from . import experiments
        if self.last_refresh is None or time.time()-self.last_refresh > 2.:
            raise ValueError('no fresh observed scene for a task proposal')
        snapshot = copy.copy(self.scene)
        snapshot.state = copy.deepcopy(self.scene.state)
        mode = payload.get('mode','build_tower')
        options = {key:value for key,value in payload.items() if key!='mode'}
        if mode=='run_sequence':
            return {'mode':mode,'sequence':options.get('sequence',['sort_colors','empty_box','build_tower','unstack_tower','reset_scene']),
                    'physical_authorized':False,'scene_version':snapshot.state.version,'reobserve_between_stages':True}
        if mode in self.EXTRA_MODES:
            context = experiments.prepare(snapshot,mode,options)
            step = experiments.next_step(snapshot,mode,context)
        else:
            step = tasks.next_transfer(mode,snapshot,options.get('object_id') or self.selected,options.get('destination'))
        proposal = {'mode':mode,'scene_version':snapshot.state.version,'physical_authorized':False,
                    'metrology_qualified':snapshot.state.evidence.get('coverage_verified',False),'step':None}
        if step is not None:
            object_id,placement = step
            proposal['step'] = {'object_id':object_id,'object_pose_m':placement.object_pose.tolist(),
                'support_surface':placement.support_surface,'support_object_id':placement.support_object_id,
                'tolerance_m':placement.tolerance,'target_px':self.geometry.project(placement.object_pose[:3,3][None])[0].tolist()}
        self.metrics['last_task_proposal'] = proposal
        return proposal

    def status(self):
        reasons = self.blockers()
        audit = None if self.robot.last_state is None else self.robot.audit_state(self.robot.last_state)
        physical_blocker = None
        if audit is not None and not audit['planning_valid']:
            physical_blocker = {'code':'UNQUALIFIED_FOLDED_START','automatic_exit_authorized':False,
                'clearance_m':audit['minimum_model_clearance_m'],'required_clearance_m':audit['required_clearance_m'],
                'required_action':'При выключенном моменте вручную вывести предплечье из сложенной позы; запястье не поворачивать, камеру и кабель держать свободными. Затем убрать руки из рабочей области.',
                'recheck':'Свежие энкодеры и штатная проверка всего пути; сброс STOP сам по себе движение не разрешает.'}
        return {'running':self.running,'mode':self.mode,'dry':None,'started':self.started,'finished':self.finished,
            'phase':self.phase,'project_detail':self.detail,'project_state':self.phase,'detail':self.detail,
            'ready':self.initialized and not reasons,'blockers':reasons,'paused':self.pause_event.is_set(),
            'stop_latched':self.stop_event.is_set(),'hardware_fault_latched':self.fault_latched,'log':list(self.log),'exit_code':0 if self.phase=='completed' else None,
            'metrics':self.metrics,'selected':self.selected,'scene_version':self.scene.state.version,
            'robot_audit':audit,'physical_blocker':physical_blocker,
            'physical_relocate_verified':any(not isinstance(r,dict) and r.success and r.evidence.get('physical_verified') for r in self.results),'experiments':{m:'blocked_pending_relocate_qualification' for m in self.MODES},
            'results':[r if isinstance(r,dict) else {'success':r.success,'reason':r.reason,'evidence':r.evidence} for r in self.results]}

    def scene_public(self):
        data = self.scene.public()
        data['observation_valid'] = self.camera_error is None and self.last_refresh is not None and time.time()-self.last_refresh <= 2.
        data['observation_error'] = self.camera_error
        if not data['observation_valid']:
            data['inside'] = data['outside'] = None
        return data

    def persist(self):
        target = self.output/'status.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.status(),ensure_ascii=False,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)))
        temporary.replace(target)

    def loop(self):
        self.initialize()
        last_vision = 0.
        last_persist = 0.
        while True:
            try:
                payload = self.commands.get(timeout=.04)
            except queue.Empty:
                try:
                    if self.robot.arm is not None:
                        state = self.robot.read_state()
                        self.cameras.record_state(state['timestamp'],state['q'],state['gripper'])
                    if time.monotonic()-last_vision >= self.config['poll_interval_s']:
                        self.refresh()
                        last_vision = time.monotonic()
                    native = self.cameras.json('/api/status')
                    state = native.get('state')
                    if state=='EMERGENCY_STOP':
                        self.native_stop(native.get('updated_at'))
                    self.last_native_state = state
                except Exception as exc:
                    self.camera_error = str(exc)
                    if self.robot.arm is not None and self.robot.arm.torque and not self.stop_event.is_set():
                        self.stop(reason='feedback/watchdog: '+str(exc),hardware_fault=True)
                if time.monotonic()-last_persist > 2.:
                    self.persist()
                    last_persist = time.monotonic()
            else:
                if payload.get('control')=='reset':
                    try:
                        self.perform_reset()
                    except Exception as exc:
                        self.stop(reason='reset refused: '+str(exc),forward=False)
                else:
                    self.execute(payload)
