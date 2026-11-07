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
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.pending_grasp_file = self.output/'pending-supported-grasp.json'
        self.pending_grasp = json.loads(self.pending_grasp_file.read_text()) if self.pending_grasp_file.exists() else None
        self.geometry = Geometry(self.config['native_root'])
        self.cameras = Cameras(self.config['api'],self.config['frame_max_age_s'],self.config.get('camera_controls'))
        self.robot = Robot(self.config['native_root'],self.output/'controller')
        self.robot.health_interval_s = self.config['poll_interval_s']
        self.scene = Scene(self.geometry,self.robot)
        attachment_path = self.root/self.config['attachment_file']
        if attachment_path.exists():
            self.scene.attachment_hypothesis = json.loads(attachment_path.read_text())
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
        self.metrics = {key:None for key in ('first_command_s','vision_s','grasp_planning_s','ik_s',
            'collision_checking_s','motion_planning_s','actual_motion_s','verification_s')}
        self.results = []
        self.motion.metrics = self.metrics
        self.motion.planner.timings = self.metrics
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
        self.cameras_initialized = False
        self.camera_retry_at = 0.
        self.scene_error = None
        self.scene_lock = self.scene.lock
        self.preparing_scene = False
        self.parked = False
        self.park_qualified_file = self.output/'park-qualified.json'
        self.last_motion_unix = time.time()
        for path in sorted((self.output/'controller').glob('wristlab-*.log'),key=lambda p:p.stat().st_mtime,reverse=True):
            moved = [line for line in path.read_text().splitlines() if ' --- ' in line]
            if moved:
                self.last_motion_unix = float(moved[-1].split()[0])
                break
        self.motor_warnings = []
        self.park_error = None
        self.park_preview = None
        self.idle_park_attempted_at = 0.
        # A physical proposal belongs to this executor instance, even when
        # the old report is still on disk for diagnosis.
        self.recovery_proposal = None
        self.overview_preview = None
        self.physical_progress = {}
        progress = self.output/'physical-progress.json'
        if progress.exists():
            self.physical_progress = json.loads(progress.read_text())
        self.action_measurements = deque(maxlen=80)
        journal=self.output/'action-measurements.jsonl'
        if journal.exists():
            with journal.open() as handle:
                for line in handle:
                    try:
                        record=json.loads(line)
                        if record.get('calibration_sha256')==self.geometry.fingerprint:
                            self.action_measurements.append(record)
                    except ValueError:
                        pass
        self.hardware_error = None
        try:
            self.known_working_configuration = json.loads((self.output/'model-audit.json').read_text()).get('known_working_configuration')
        except (OSError,ValueError):
            self.known_working_configuration = None
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
        self.initialize_cameras()
        self.initialized = True
        reasons = self.blockers()
        if not reasons:
            self.ensure_driver()
        self.report('blocked' if reasons else 'ready','; '.join(reasons) if reasons else 'Ready')
        self.persist()

    def initialize_cameras(self):
        try:
            if 'camera_controls' not in self.metrics:
                self.metrics['camera_controls'] = self.cameras.apply_controls()
            self.cameras.on_native_stop = self.native_stop
            self.cameras.start_capture()
            # The line-only detector confused table/cable edges with boxes in
            # this scene. Reuse the already-loaded native model once, before
            # Ready; experiments only track the resulting current observation.
            if not self.scene.boxes_semantically_seeded:
                box_started = time.perf_counter()
                scan = self.cameras.json('/api/scan-hybrid',{},timeout=30.)
                with self.scene_lock:
                    self.scene.install_box_proposals(scan)
                (self.output/'box-observation.json').write_text(json.dumps(scan))
                self.metrics['box_detection_startup_s'] = time.perf_counter()-box_started
            wrist_path = self.output/'wrist-camera-calibration.json'
            if wrist_path.exists():
                self.geometry.install_wrist_calibration(json.loads(wrist_path.read_text()),self.robot)
            self.refresh(measure_attachment=True)
            self.cameras_initialized = True
            self.cameras.on_frames = self.background_refresh
        except Exception as exc:
            self.camera_error = str(exc)
            self.camera_retry_at = time.monotonic()+2.

    def read_release_checkpoint(self):
        import pickle
        path = self.output/'released-placement.pkl'
        # A power cycle can move the arm and the scene while the runtime is
        # absent. Historical release geometry cannot authorize a new retreat.
        with path.open('rb') as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get('boot_id')!=self.boot_id:
            raise ValueError('released placement is not from this boot; observe a new scene')
        return checkpoint

    def refresh(self, measure_attachment=False):
        begin = time.monotonic()
        self.scene_lock.acquire()
        stage = 'frames'
        try:
            frames = self.cameras.latest_pair()
            t = time.monotonic()
            from .vision import frame_quality
            qualities = {name:frame_quality(frame,40. if name=='overview' else 15.) for name,frame in frames.items()}
            self.metrics['camera_quality'] = qualities
            invalid = [name+': '+q['reason'] for name,q in qualities.items() if not q['usable']]
            if invalid:
                raise ValueError('; '.join(invalid))
            self.camera_error = None
            stage = 'detection'
            self.scene.update_scene(frames,measure_attachment=measure_attachment and not self.running)
            self.metrics['vision_s'] = time.monotonic()-t
            self.metrics['camera_fetch_s'] = t-begin
            self.scene_error = None
            self.frame_history.append(dict(frames))
            self.last_refresh = time.time()
        except Exception as exc:
            if stage=='frames':
                self.camera_error = str(exc)
            else:
                self.scene_error = str(exc)
            raise
        finally:
            self.scene_lock.release()

    def background_refresh(self):
        if self.initialized and not self.preparing_scene:
            try:
                self.refresh()
            except Exception:
                pass  # refresh records transport and detection errors separately.

    def blockers(self):
        return [item['reason'] for item in self.blocking_conditions()]

    def blocking_conditions(self):
        reasons = self.motion.qualification_blockers(self.geometry)
        def add(code,category,reason,condition):
            reasons.append({'code':code,'category':category,'reason':reason,'clear_when':condition})
        if self.camera_error:
            add('CAMERA_FEEDBACK','FEEDBACK',self.camera_error,'обе камеры дают свежие кадры с допустимым качеством')
        if self.scene_error:
            add('SCENE_PROCESSING','PERCEPTION',self.scene_error,'детекция и обновление сцены завершены без ошибки')
        if self.last_refresh is None or time.time()-self.last_refresh > 2.:
            add('SCENE_STALE','FEEDBACK','нет актуального наблюдения сцены','сцена обновлена по свежим кадрам не более 2 с назад')
        if not self.scene.state.boxes:
            add('BOX_MISSING','GEOMETRY','коробки не обнаружены','актуальное наблюдение содержит целевую коробку')
        elif any(not (b.evidence.get('metric_qualified') or b.evidence.get('collision_qualified')) for b in self.scene.state.boxes.values()):
            add('BOX_GEOMETRY','QUALIFICATION','геометрия дна/бортов коробок не квалифицирована','все видимые коробки имеют текущий метрический обзор дна и стенок')
        if self.scene.state.obstacles:
            add('SCENE_OBSTACLES','GEOMETRY','есть объекты без подтверждённой геометрии: '+
                ', '.join(sorted({o.get('color','unknown') for o in self.scene.state.obstacles})),
                'актуальные наблюдения подтверждают положение и размеры всех видимых препятствий')
        if self.scene.held is not None and not self.running:
            add('PAYLOAD_HELD','RECOVERY','после ошибки планирования предмет удерживается',
                'предмет установлен по отдельно проверенному плану восстановления; автоматическое раскрытие запрещено')
        if self.pending_grasp is not None and self.scene.held is None:
            add('HELD_RECOVERY_PENDING','RECOVERY','сохранён поддерживаемый захват; требуется продолжение этого переноса',
                'проверены фактические углы и усилие, восстановлена привязка удерживаемого кубика')
        if self.stop_event.is_set():
            add('STOP_LATCHED','STOP','STOP зафиксирован','явный сброс STOP при простое; новый Start и все проверки безопасности всё равно обязательны')
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
        # A native tick follows the actual goal write, including a tick that
        # fails the camera/geometry guard. Preserve that physical fact on error.
        if self.robot.first_command_at is None:
            self.robot.first_command_at = time.monotonic()
        self.check_cancel()
        self.cameras.check()
        self.robot.read_motor_health()
        self.motion.guarded_tick(record)
        stamp = time.time()
        self.robot.last_state = {'timestamp':stamp,'positions':dict(record['pos']),
            'loads':dict(record['load']),'currents':dict(record['ma']),
            'q':self.robot.from_native(record['pos']),'gripper':record['pos']['gripper']}
        self.last_motion_unix = stamp
        self.parked = False
        self.monitor_motors()
        self.cameras.record_state(stamp,self.robot.from_native(record['pos']),record['pos']['gripper'])

    def monitor_motors(self):
        warnings,critical = [],[]
        state = self.robot.last_state
        if state is None or self.robot.arm is None:
            return
        for name,motor in self.robot.motor_health.items():
            limits = motor['limits']
            if limits['Max_Temperature_Limit'] and motor['temperature_c']>=limits['Max_Temperature_Limit']:
                critical.append(name+': temperature at configured servo limit')
            if motor['status_raw']:
                critical.append(name+': drive status '+hex(motor['status_raw']))
            if ((limits['Max_Voltage_Limit'] and motor['voltage_v']>limits['Max_Voltage_Limit']/10.) or
                (limits['Min_Voltage_Limit'] and motor['voltage_v']<limits['Min_Voltage_Limit']/10.)):
                critical.append(name+': voltage outside configured servo limits')
            load_limit = min(self.robot.arm.raw.load_max,limits['Overload_Torque'] or self.robot.arm.raw.load_max)
            if abs(state['loads'].get(name,0.))>=load_limit:
                critical.append(name+': native load limit exceeded')
            if limits['Protection_Current'] and abs(motor['current_ma'])>=limits['Protection_Current']*6.5:
                critical.append(name+': configured servo current limit exceeded')
            span = self.robot.backend.model.gripper_span_deg/100. if name=='gripper' else 1.
            if abs(motor['goal']-state['positions'][name])*span>self.robot.arm.raw.lag_max:
                warnings.append(name+': following error exceeds native limit')
        holding = any(m['torque_enabled'] for m in self.robot.motor_health.values())
        if holding and not self.running and not self.parked and time.time()-self.last_motion_unix>=self.config['hold_warning_after_s']:
            warnings.append('Длительное удержание: '+('Park запрещён при STOP; ' if self.stop_event.is_set() else 'требуется безопасный Park; ')+
                            'не отключайте момент без поддержки руки')
        self.motor_warnings = critical+warnings
        if critical and not self.stop_event.is_set():
            self.stop(reason='motor protection: '+'; '.join(critical),hardware_fault=True)

    def prepare(self, payload):
        with self.submit_lock:
            if self.running or not self.commands.empty():
                raise ValueError('another task is running or queued')
            preview_recovery = payload.get('mode')=='RECOVERY_FROM_INVALID_START' and payload.get('dry') is True
            preview_folded = payload.get('mode')=='FOLDED_STARTUP' and payload.get('dry') is True
            preview_view = payload.get('mode')=='OVERVIEW_CLEARANCE' and payload.get('dry') is True
            if not (preview_recovery or preview_folded or preview_view):
                self.check_cancel()
            if not self.initialized or (self.scene.held is not None and not payload.get('resume_supported_grasp')):
                raise ValueError('startup requires initialized feedback and an empty hand')
            if not (preview_recovery or preview_folded or preview_view) and payload.get('cable_and_hidden_surfaces_clear') is not True:
                raise ValueError('operator observation is required for this attended startup')
            self.running = True
            self.commands.put_nowait(dict(payload,control='prepare'))

    def perform_prepare(self, payload):
        self.preparing_scene = True
        try:
            self.started = time.monotonic()
            self.robot.first_command_at = None
            if payload.get('mode')=='OVERVIEW_CLEARANCE':
                self.results = [self.prepare_overview_clearance(payload)]
                return
            if payload.get('mode')=='FOLDED_STARTUP' and payload.get('dry') is True:
                generation = self.stop_generation
                def check_preview():
                    if generation!=self.stop_generation or not self.robot.owns_lease():
                        raise RuntimeError('new STOP or hardware ownership lost during folded-start preview')
                    self.cameras.check()
                check_preview()
                self.refresh()
                self.results = [self.motion.prepare_ready(self.scene,self.geometry,{},dry=True,planning_check=check_preview)]
                self.report('checked','folded exit geometry checked; STOP retained; no motion')
                return
            if payload.get('mode')=='RECOVERY_FROM_INVALID_START':
                self.started = time.monotonic()
                self.robot.first_command_at = None
                self.results = [self.preview_invalid_start() if payload.get('dry') is True else self.execute_invalid_start(payload)]
                return
            self.check_cancel()
            self.cameras.check()
            self.refresh()
            attendance = {'cable_and_hidden_surfaces_clear':True,'timestamp':time.time(),
                          'source':payload.get('observation_source','operator'),
                          'scope':'attended overhead pick/place; cable observed by operator' if payload.get('grasp_mode')=='overhead_grasp' else
                                  'initial folded exit only; cable is observed by operator, not reconstructed'}
            self.started = time.monotonic()
            self.robot.first_command_at = None
            if payload.get('grasp_mode')=='overhead_grasp':
                if payload.get('resume_released_placement'):
                    self.preparing_scene = False
                    self.results = [self.resume_released_placement(attendance)]
                    return
                recovery = bool(payload.get('recover_empty_approach') or payload.get('resume_supported_grasp'))
                result = {} if payload.get('resume_supported_grasp') and self.scene.held is not None else self.prepare_overhead(attendance,recovery=recovery)
                if payload.get('resume_supported_grasp'):
                    self.preparing_scene = False
                    self.results = [self.resume_supported_grasp()]
                    return
                if payload.get('recover_empty_approach'):
                    import pickle
                    with (self.output/'last-relocate-plan.pkl').open('rb') as handle:
                        plan = pickle.load(handle)
                    selected = next(o.id for o in self.scene.state.objects.values() if o.color=='green')
                    placement = tasks.placement_in_box(self.scene,selected,'box-1')
                    native = self.scene.native_scene(selected,placement)
                    self.motion.qualification_blockers(self.geometry)
                    world = self.motion.make_world(native)
                    self.preparing_scene = False
                    result['returned'] = self.motion.recover_empty_approach(plan,world)
            else:
                result = self.motion.prepare_ready(self.scene,self.geometry,attendance,bool(payload.get('dry')))
            self.results = [result]
            self.refresh()
            self.report('checked' if payload.get('dry') else 'prepared',
                        'checked empty-hand return physically completed' if payload.get('recover_empty_approach') else
                        'overhead geometry installed; no motion' if payload.get('grasp_mode')=='overhead_grasp' else
                        'startup path checked; no motion' if payload.get('dry') else 'folded exit physically completed; ordinary margins satisfied')
        except Exception as exc:
            self.report('failed',str(exc))
            self.results = [{'success':False,'reason':str(exc),'physical_verified':False}]
            held_planning = isinstance(exc,ValueError) and self.scene.held is not None and self.motion.driver is not None and self.motion.driver.retention.armed
            if held_planning:
                self.motion.driver.confirm_retention()
                self.report('recovering',str(exc))
            elif self.robot.first_command_at is not None and (payload.get('grasp_mode')!='overhead_grasp' or payload.get('recover_empty_approach') or payload.get('resume_supported_grasp') or payload.get('resume_released_placement')):
                self.stop(reason='startup: '+str(exc),hardware_fault=True)
        finally:
            if self.robot.first_command_at is not None:
                self.metrics['first_command_s'] = self.robot.first_command_at-self.started
            self.running = False
            self.preparing_scene = False
            self.persist()

    def prepare_overview_clearance(self,payload):
        dry = payload.get('dry') is True
        previous = self.overview_preview
        generation = self.stop_generation
        if not dry and (previous is None or payload.get('operator_at_stop') is not True or
                        payload.get('proposal_checked_at')!=previous['checked_at'] or generation!=previous['stop_generation']):
            raise ValueError('a current checked overview route and explicit attended approval are required')
        def check():
            if generation!=self.stop_generation or not self.robot.owns_lease():
                raise RuntimeError('STOP or hardware ownership changed during overview check')
            if not dry:
                self.check_cancel()
            self.cameras.check()
        check()
        self.refresh()
        if not self.scene.state.boxes:
            raise ValueError('no current box outlines for a checked view change')
        self.report('checking','noncontact overview clearance from fresh geometry')
        path,world,mapping = self.motion.overview_clear_path(self.scene,self.geometry,check)
        proposal = {'start':path[0],'goal':path[-1],'checked_at':time.time(),'stop_generation':generation,
                    'partial_cube_clearance_m':world.partial_clearance.copy(),
                    'holding_hardware_goals':mapping['hardware_goals'],'stop_policy':mapping['supported_stop_policy'],
                    'success':True,'physical_verified':False,'actuator_commands_sent':False}
        if dry:
            self.overview_preview = proposal
            self.report('checked','overview view change checked; no actuator commands')
            return proposal
        if any(previous[key]!=proposal[key] for key in ('start','goal','holding_hardware_goals')):
            raise ValueError('the approved overview route or holding goals changed; recalculate before execution')
        check()
        self.motion.startup_world = world
        self.motion.executing_recovery = True
        self.motion.recovery_mapping = mapping
        try:
            travel = abs(path[-1]['shoulder_pan']-path[0]['shoulder_pan'])
            self.report('moving','checked noncontact overview view')
            with self.robot.synchronized_follow(mapping):
                self.robot.arm.follow(path,duration=max(1.,1.875*travel/2.),settle=1.,within=.7,lead=False,step='overview_clearance')
        finally:
            self.motion.startup_world = None
            self.motion.executing_recovery = False
        final = self.robot.read_state()['positions']
        reason = world.reason(final)
        if reason or abs(final['shoulder_pan']-path[-1]['shoulder_pan'])>self.motion.planner.tracking_error_deg:
            raise RuntimeError('overview move endpoint: '+str(reason or 'tracking error'))
        self.refresh()
        self.overview_preview = None
        self.report('prepared','overview view reached; '+('obstacles remain unresolved' if self.scene.state.obstacles else 'all visible cubes localized'))
        return proposal|{'actual':final,'physical_verified':True,'actuator_commands_sent':True,
                         'remaining_obstacles':self.scene.state.obstacles}

    def preview_invalid_start(self):
        """Read-only recovery diagnosis is permitted with the STOP latch set."""
        generation = self.stop_generation
        last_feedback = 0.
        def planning_cancel():
            nonlocal last_feedback
            if generation!=self.stop_generation or not self.robot.owns_lease():
                raise RuntimeError('new STOP or lost hardware ownership during recovery calculation')
            self.cameras.check()
            if time.monotonic()-last_feedback>=self.config['poll_interval_s']:
                current = self.robot.read_state()
                self.cameras.record_state(current['timestamp'],current['q'],current['gripper'])
                self.monitor_motors()
                last_feedback = time.monotonic()
        planning_cancel()
        saved = self.read_release_checkpoint()
        if saved['calibration_sha256']!=self.geometry.fingerprint or self.scene.held is not None:
            raise ValueError('no matching released-cube checkpoint')
        state = self.robot.read_state()
        if not self.robot.arm.torque or not all(m['torque_enabled'] for m in state['motor_health'].values()):
            raise ValueError('retreat recovery requires an uninterrupted powered hold')
        if (abs(state['loads']['gripper']) or abs(state['currents']['gripper']) or
            state['gripper']<saved['release_actual']['gripper']-self.robot.encoder_steps()['gripper']):
            raise ValueError('empty jaws are not confirmed by current feedback')
        attendance = json.loads((self.output/'startup-trial.json').read_text())['attendance']
        self.motion.grasp_mode = self.geometry.grasp_mode = 'overhead_grasp'
        self.motion.operator_observation = attendance
        self.motion.qualification_blockers(self.geometry)
        self.motion.planner.command_sag = float(self.robot.arm.raw.sag)
        placed = next(p.placed_cube for p in reversed(saved['plan'].phases) if p.placed_cube is not None)
        self.report('checking','retreat recovery from fresh encoders; STOP retained; no motor commands')
        self.preparing_scene = False
        begin = time.perf_counter()
        try:
            _,_,_,report = self.motion.plan_invalid_start(saved['native_scene'],state,placed,planning_cancel)
        except (ValueError,RuntimeError) as exc:
            report = (self.motion.recovery_report or {})|{'status':'BLOCKED','reason':str(exc)}
        final = self.robot.read_state()
        if any(abs(final['positions'][j]-state['positions'][j])>self.robot.encoder_steps()[j] for j in state['positions']):
            report.update(status='BLOCKED',reason='encoders changed during read-only calculation; recalculate from actual state')
        report.update(feedback_timestamp=state['timestamp'],checked_at=time.time(),
            planning_s=time.perf_counter()-begin,stop_latched=self.stop_event.is_set(),
            stop_generation=self.stop_generation,calibration_sha256=self.geometry.fingerprint,
            attachment_sha256=None if self.motion.attachment is None else self.motion.attachment.fingerprint,
            success=False,physical_verified=False,actuator_commands_sent=False)
        self.recovery_proposal = report
        (self.output/'invalid-start-proposal.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        self.report('blocked' if report['status']=='BLOCKED' else 'checked',report.get('reason',report['status']))
        return report

    def execute_invalid_start(self, payload):
        """Execute only a specifically approved, complete recovery proposal."""
        from dataclasses import replace
        from so101_pick_place.experiments.box_extraction.planner import Phase
        self.check_cancel()
        self.cameras.check()
        proposal = self.recovery_proposal or {}
        if (payload.get('operator_at_stop') is not True or
            payload.get('proposal_checked_at')!=proposal.get('checked_at') or
            proposal.get('status')!='READY_FOR_ATTENDED_CONFIRMATION'):
            raise ValueError('a complete checked Retreat/Park and explicit attended approval are required')
        if proposal['stop_generation']!=self.stop_generation or proposal['calibration_sha256']!=self.geometry.fingerprint:
            raise ValueError('STOP or calibration changed after the recovery calculation')
        saved = self.read_release_checkpoint()
        state = self.robot.read_state()
        self.robot.read_motor_health(force=True)
        state['motor_health'] = self.robot.motor_health
        if (self.scene.held is not None or abs(state['loads']['gripper']) or abs(state['currents']['gripper'])):
            raise ValueError('recovery requires confirmed empty jaws')
        synchronization = self.robot.synchronize_reference(state,self.robot.arm.raw.sag)
        if (any(state['positions'][j]!=proposal['start'][j] for j in state['positions']) or
            synchronization['hardware_goals']!=proposal['holding_hardware_goals']):
            raise ValueError('actual encoders or powered holding goals changed; a fresh proposal is required')
        self.geometry.grasp_mode = self.motion.grasp_mode = 'overhead_grasp'
        self.motion.operator_observation = json.loads((self.output/'startup-trial.json').read_text())['attendance']
        self.motion.qualification_blockers(self.geometry)
        if self.motion.attachment.fingerprint!=proposal['attachment_sha256']:
            raise ValueError('camera/mount geometry changed after the recovery calculation')
        placed = next(p.placed_cube for p in reversed(saved['plan'].phases) if p.placed_cube is not None)
        world = self.motion.recovery_world(saved['native_scene'],state['positions'])
        self.motion.install_recovery_bounds(world,proposal['initial_gaps_with_execution_error_m'])
        normal = self.motion.make_world(saved['native_scene'])
        self.motion.recovery_mapping = synchronization
        self.motion.recovery_report = proposal
        self.motion.prepare_driver(self.cameras.check)
        trial = json.loads((self.output/'contact-free-trial.json').read_text())
        if not trial.get('passed') or trial['attachment_sha256']!=self.motion.attachment.fingerprint:
            raise ValueError('matching contact-free attachment evidence is missing')
        self.motion.contact_free_passed = True
        self.motion.trial_attachment_sha256 = self.motion.attachment.fingerprint
        self.scene.state.boxes = saved['boxes']
        self.scene.metric_boxes = saved['boxes'].copy()
        self.scene.boxes_semantically_seeded = True
        self.scene.state.objects[saved['object'].id] = saved['object']
        self.scene.track_memory[saved['object'].id] = saved['object']
        self.scene.metric_objects.update(saved['native_scene'].cubes)
        self.scene.pending_placement = saved['object'].id,saved['object'].color
        self.preparing_scene = False
        self.refresh()
        phase = Phase('RECOVERY_FROM_INVALID_START',tuple(proposal['joint_path']),jaw_contact=True,placed_cube=placed)
        plan = replace(saved['plan'],phases=(phase,))
        self.robot.enable()
        self.motion.execute_trajectory(phase,plan,world)
        actual = self.robot.read_state()['positions']
        refusal = normal.reason(actual,target=plan.target_id,placed_cube=placed)
        frames = self.robot.backend.model.frames(actual)
        heights = [self.robot.backend._transform_points(frames[link],self.robot.backend.meshes[link])[:,2].min()
                   for link in ('gripper_link','moving_jaw_so101_v1_link')]
        heights.extend(self.robot.backend._transform_points(frames[parent],vertices)[:,2].min()
                       for name,parent,lo,hi,vertices in self.motion.attachment.parts)
        rim = float((saved['native_scene'].box.base_from_box@np.r_[0.,0.,saved['native_scene'].box.obstacle_top_m,1.])[2])
        if refusal or min(heights)<rim+self.robot.backend.model.gap:
            raise RuntimeError('retreat endpoint: '+str(refusal or 'tool is not completely above the box rim'))
        self.physical_progress.update(retreat_completed=True,retreat_actual=actual,retreat_at=time.time())
        (self.output/'physical-progress.json').write_text(json.dumps(self.physical_progress,indent=2))
        # Rebase from the newly measured endpoint; never jump back to a stale
        # control reference at the boundary between retreat and Park.
        state = self.robot.read_state()
        self.robot.read_motor_health(force=True)
        state['motor_health'] = self.robot.motor_health
        self.motion.recovery_mapping = self.robot.synchronize_reference(state,self.robot.arm.raw.sag)
        old_command = self.motion.planner._command_pose
        self.motion.planner._command_pose = lambda q:self.robot.synchronized_command(q,self.motion.recovery_mapping)
        try:
            park = self.motion.plan_park(state['positions'],saved['plan'],normal,recovery=True)
            self.motion.execute_trajectory(park,saved['plan'],normal)
        finally:
            self.motion.planner._command_pose = old_command
        final = self.robot.read_state()['positions']
        refusal = normal.reason(final,target=plan.target_id,placed_cube=placed)
        if refusal or max(abs(final[j]-park.joints[-1][j]) for j in final if j!='gripper')>self.motion.planner.tracking_error_deg:
            raise RuntimeError('Park endpoint: '+str(refusal or 'following error'))
        self.parked = True
        self.physical_progress.update(park_completed=True,park_actual=final,park_at=time.time())
        (self.output/'physical-progress.json').write_text(json.dumps(self.physical_progress,indent=2))
        self.motion.plan,self.motion.world = saved['plan'],normal
        self.park_qualified_file.write_text(json.dumps({'passed':True,'path':park.joints,
            'calibration_sha256':self.geometry.fingerprint,'attachment_sha256':self.motion.attachment.fingerprint,
            'actual':final,'checked_at':time.time(),'attended':True},indent=2))
        verified,pose = self.actions.verify_placement(saved['object'],saved['placement'])
        if verified:
            self.scene.detach_object(plan.target_id,pose)
        self.report('completed','Retreat выполнен; PARKED; '+('зелёный кубик подтверждён камерой' if verified else 'метрическая проверка кубика пока не подтверждена'))
        return {'success':True,'physical_verified':True,'retreat_completed':True,'parked':True,
                'actual':final,'placement_verified':verified,'stop_reset':False}

    def park_arm(self, previous=None, world=None, *, dry=False, attended=False, expected_goal=None):
        """Shared Park entry: the STOP latch is checked before any planning or motion."""
        self.check_cancel()
        self.cameras.check()
        if self.scene.held is not None:
            raise ValueError('Park: object held; finish a separately checked relocate to a safe support before parking')
        if previous is None and self.motion.plan is not None and self.motion.world is not None:
            previous,world = self.motion.plan,self.motion.world
        if previous is None:
            path = self.output/'released-placement.pkl'
            if not path.exists():
                raise ValueError('Park has no verified current manipulation scene')
            saved = self.read_release_checkpoint()
            if saved['calibration_sha256']!=self.geometry.fingerprint:
                raise ValueError('Park checkpoint calibration differs')
            previous = saved['plan']
            world = self.motion.make_world(saved['native_scene'])
        state = self.robot.read_state()
        phase = self.motion.plan_park(state['positions'],previous,world)
        if expected_goal is not None and any(abs(expected_goal[j]-phase.joints[-1][j])>self.robot.encoder_steps()[j] for j in expected_goal):
            raise ValueError('Park destination changed; show a new preview before execution')
        proposal = {'status':'PARK_CHECKED','start':state['positions'],'goal':phase.joints[-1],
            'actuator_commands_sent':False,'requires_attended_trial':not self.park_qualified_file.exists(),
            'success':True,'reason':'Park path checked; no actuator commands','physical_verified':False}
        if dry:
            self.park_preview = proposal|{'stop_generation':self.stop_generation}
            return proposal
        if not self.park_qualified_file.exists() and not attended:
            raise ValueError('first Park transition requires an attended execution after path preview')
        if not attended:
            qualified = json.loads(self.park_qualified_file.read_text())
            if (qualified.get('calibration_sha256')!=self.geometry.fingerprint or
                qualified.get('attachment_sha256')!=self.motion.attachment.fingerprint):
                raise ValueError('Park qualification differs from current geometry')
            reference = qualified['path']
            steps = self.robot.encoder_steps()
            if any(min(max(abs(q[j]-r[j])/steps[j] for j in q if j!='gripper') for r in reference)>1. for q in phase.joints):
                raise ValueError('new Park route needs an attended cable sweep; outside the physically checked route')
        self.ensure_driver()
        trial = json.loads((self.output/'contact-free-trial.json').read_text())
        if not trial.get('passed') or trial.get('attachment_sha256')!=self.motion.attachment.fingerprint:
            raise ValueError('Park: matching physical camera-attachment evidence is missing')
        self.motion.contact_free_passed = True
        self.motion.trial_attachment_sha256 = self.motion.attachment.fingerprint
        self.robot.enable()
        try:
            result = self.motion.park_arm(previous,world,phase)
        except Exception as exc:
            self.stop(reason='Park execution: '+str(exc),hardware_fault=True)
            raise
        result.update(success=True,reason='Park position confirmed by encoders',physical_verified=True)
        self.parked = True
        self.park_error = None
        if attended:
            self.park_qualified_file.write_text(json.dumps(result|{'calibration_sha256':self.geometry.fingerprint,
                'attachment_sha256':self.motion.attachment.fingerprint,'path':phase.joints,'checked_at':time.time()},indent=2))
        self.report('parked','PARKED: положение подтверждено энкодерами; момент сохранён')
        return result

    def request_park(self, payload):
        with self.submit_lock:
            if self.running or not self.commands.empty():
                raise ValueError('another task is running or queued')
            self.check_cancel()
            if not payload.get('dry',True):
                if (payload.get('operator_at_stop') is not True or self.park_preview is None or
                    self.park_preview['stop_generation']!=self.stop_generation):
                    raise ValueError('Park needs a current preview and explicit operator-at-STOP confirmation')
                payload = payload|{'expected_goal':self.park_preview['goal']}
                self.park_preview = None
            self.running = True
            self.commands.put_nowait(dict(payload,control='park'))

    def perform_park(self, payload):
        moved = self.robot.first_command_at
        try:
            result = self.park_arm(dry=bool(payload.get('dry',True)),attended=payload.get('operator_at_stop') is True,
                                   expected_goal=payload.get('expected_goal'))
            self.results = [result]
            if payload.get('dry',True):
                self.report('checked',json.dumps(result,ensure_ascii=False))
        except Exception as exc:
            self.park_error = str(exc)
            if self.robot.first_command_at!=moved:
                self.stop(reason='Park: '+str(exc),hardware_fault=True)
            else:
                self.report('blocked','Park: '+str(exc))
            self.results = [{'success':False,'reason':str(exc),'physical_verified':False}]
        finally:
            self.running = False
            self.persist()

    def measure_overview(self, stationary):
        """Take independent native measurements while the owner monitors the held pose."""
        result = queue.Queue(maxsize=1)
        cancelled = threading.Event()
        started = time.time()

        def capture():
            try:
                sources = []
                for _ in range(3):
                    if cancelled.is_set() or self.stop_event.is_set():
                        raise RuntimeError('overview measurement cancelled')
                    sources.append(self.cameras.json('/api/scan-hybrid',{},timeout=30.))
                result.put(sources)
            except Exception as exc:
                result.put(exc)

        threading.Thread(target=capture,name='wristlab-overview-measurement',daemon=True).start()
        steps = self.robot.encoder_steps()
        try:
            while True:
                self.check_cancel()
                self.cameras.check()
                state = self.robot.read_state()
                if any(abs(state['positions'][j]-stationary[j])>steps[j] for j in stationary):
                    raise ValueError('arm changed pose during overview measurement')
                self.cameras.record_state(state['timestamp'],state['q'],state['gripper'])
                self.monitor_motors()
                self.check_cancel()
                try:
                    sources = result.get(timeout=self.config['poll_interval_s'])
                    break
                except queue.Empty:
                    pass
        finally:
            cancelled.set()
        if isinstance(sources,Exception):
            raise sources
        self.check_cancel()
        frames = [source['frame'] for source in sources]
        if (len({f['sequence'] for f in frames})!=3 or
            min(f['captured_at_unix_s'] for f in frames)<started-self.cameras.max_age):
            raise ValueError('overview measurements did not come from three fresh distinct frames')
        return sources

    def prepare_overhead(self, attendance, recovery=False):
        """Restore the successful overview pipeline, without a wrist transform."""
        from box_extract.scene_builder import SceneParameters,measure_box,spatial_cube,merge_cubes
        from box_extract.opening_fallback import estimate_opening,OpeningTracker
        from so101_pick_place.vision.calibration_control import validate_overview_control
        self.check_cancel()
        state = self.robot.read_state()
        if not self.robot.audit_state(state)['planning_valid']:
            raise ValueError('overhead operation requires the measured working pose')
        self.refresh()
        if self.scene.state.obstacles and not recovery:
            raise ValueError('resolve visible unlocalized obstacles before measuring for Pick & Place')
        frame = self.scene.frames['overview']
        current_boxes = {key:np.asarray(box.evidence['polygon']) for key,box in self.scene.state.boxes.items()}
        if recovery:
            from .types import Frame
            from .vision import detect_cubes
            observed = {o.color:o.pose[:3,3].copy() for o in self.scene.state.objects.values()}
            reference = json.loads((self.output/'pre-approach-overview-reference.json').read_text())
            pixels = cv2.imread(str(Path(reference['artifacts'])/'frame.jpg'))
            stamp = reference['frame']['captured_at_unix_s']
            reference_frame = Frame('overview','retained-before-contact-free-approach',pixels,stamp,stamp,(stamp,stamp))
            self.geometry.grasp_mode = 'overhead_grasp'
            self.scene.update_scene({'overview':reference_frame,'wrist':self.scene.frames['wrist']})
            from .vision import color_masks
            wrist_colors = {color for color,mask in color_masks(self.scene.frames['wrist'].image).items() if np.count_nonzero(mask)>100}
            for obj in self.scene.state.objects.values():
                if obj.color in observed and np.linalg.norm(observed[obj.color]-obj.pose[:3,3])>.003:
                    raise ValueError('visible cube moved since the noncontact route was checked')
                if obj.color not in observed and obj.color not in wrist_colors:
                    raise ValueError('occluded cube is not confirmed by the wrist image')
            control = json.loads((self.output/'visual-measurement.json').read_text())['overview_fk_control']
            if control.get('passed') is not True:
                raise ValueError('the original overview control did not pass')
            control = dict(control,reused_after_self_occlusion=True,current_marker_measurement=False)
        else:
            control = validate_overview_control(frame.image,self.geometry.calibration,self.robot.backend.model,state['positions'])
        source_names = ('pre-approach-overview-reference.json','overview-confirm-0.json','overview-confirm-1.json')
        if recovery:
            sources = [json.loads((self.output/name).read_text()) for name in source_names]
        else:
            self.report('observe','measuring the current overview geometry without motion')
            sources = self.measure_overview(state['positions'])
            self.refresh()
            if self.scene.state.obstacles:
                raise ValueError('scene coverage changed during overview measurement')
            frame = self.scene.frames['overview']
            current_boxes = {key:np.asarray(box.evidence['polygon']) for key,box in self.scene.state.boxes.items()}
            control = validate_overview_control(frame.image,self.geometry.calibration,self.robot.backend.model,state['positions'])
        params = SceneParameters()
        measured = {}
        for box_id,box in self.scene.state.boxes.items():
            observations = []
            tracker = OpeningTracker()
            for scan in sources:
                image = cv2.imread(str(Path(scan['artifacts'])/'frame.jpg'))
                bodies = [d for d in scan['detections'] if d['label']=='box']
                body = min(bodies,key=lambda d:np.linalg.norm(np.asarray(d['center_px'])-box.evidence['center_px']))
                outline = cv2.convexHull(np.float32(np.vstack(body['silhouette_polygons_px'])))
                current_outline = cv2.convexHull(np.float32(current_boxes[box_id]))
                intersection = cv2.intersectConvexConvex(outline,current_outline)[0]
                union = cv2.contourArea(outline)+cv2.contourArea(current_outline)-intersection
                if intersection/max(1.,union)<(.92 if recovery else .8):
                    raise ValueError(box_id+': current box contour differs from the measured observations')
                openings = [d for d in scan['detections'] if d['label']=='box_opening' and
                            cv2.pointPolygonTest(outline,tuple(d['center_px']),False)>=0]
                subset = dict(scan,detections=[body]+openings)
                if not openings:
                    opening,detail = estimate_opening(image,subset,tracker=tracker)
                    if opening is None:
                        raise ValueError(box_id+': '+str(detail))
                    subset['detections'].append(opening)
                observations.append(measure_box(subset,self.geometry.calibration,params,len(sources)))
            centers = np.array([b.base_from_box[:2,3] for b in observations])
            if np.linalg.norm(np.ptp(centers,axis=0))>params.max_box_spread_m:
                from .scene import BoxKeepout
                polygon = np.asarray(box.evidence['polygon'])
                points = np.array([self.geometry.intersect_ray_with_plane(self.geometry.pixel_ray_in_base(p,frame),
                                  ([0.,0.,1.],-height)) for height in (0.,params.flap_top_m) for p in polygon])
                native = BoxKeepout(np.r_[points[:,:2].min(0)-.008,-.01],
                                    np.r_[points[:,:2].max(0)+.008,params.flap_top_m+.008])
                box.evidence.update(metric_qualified=False,collision_qualified=True,native_box=native,
                                    entry_allowed=False,geometry_mode='overhead_keepout',
                                    reason='opening is not repeatable; entry and placement are prohibited')
                box.pose = native.base_from_box.copy()
            else:
                native = observations[-1]
                box.evidence.update(metric_qualified=True,native_box=native,geometry_mode='overhead_grasp',
                                    support_assumptions=native.evidence['floor_assumption'])
                box.pose = native.base_from_box.copy()
                box.inner_dimensions = np.r_[native.inner_max_xy-native.inner_min_xy,native.obstacle_top_m]
                box.wall_height = native.obstacle_top_m
            measured[box_id] = box
        for obj in self.scene.state.objects.values():
            observations = []
            pixel = np.mean(obj.evidence['quad'],axis=0)
            for scan in sources:
                candidates = [d for d in scan['detections'] if d['label']=='cube' and d.get('geometry')]
                detection = min(candidates,key=lambda d:np.linalg.norm(np.asarray(d['center_px'])-pixel))
                if np.linalg.norm(np.asarray(detection['center_px'])-pixel)>5.:
                    raise ValueError(obj.id+': cube moved since full silhouette measurement')
                cube = spatial_cube(detection,params)
                if abs(cube.side_m-float(self.geometry.known_cube['cube_size_m']))>.003:
                    raise ValueError('overview silhouette disagrees with the measured cube size')
                observations.append([cube])
            cubes,spread = merge_cubes(observations,params)
            self.scene.metric_objects[obj.id] = cubes[0]
        self.scene.metric_boxes.update(measured)
        self.geometry.grasp_mode = self.motion.grasp_mode = 'overhead_grasp'
        self.motion.operator_observation = attendance
        self.refresh()
        invalid = [o.id for o in self.scene.state.objects.values() if not o.evidence.get('metric_qualified')]
        if invalid:
            raise ValueError('current silhouettes do not confirm overview geometry: '+str(invalid))
        self.scene.state.evidence['coverage_verified'] = not self.scene.state.obstacles
        self.ensure_driver()
        if not recovery:
            for name,scan in zip(source_names,sources):
                path = self.output/name
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(scan))
                temporary.replace(path)
        return {'grasp_mode':'overhead_grasp','wrist_refinement':False,'overview_control':control,
                'sources':[s['artifacts'] for s in sources],'assumptions':params.__dict__,
                'operator_observation':attendance,'physical_pick_place_completed':False}

    def resume_supported_grasp(self):
        """Same native supported regrasp used by the successful held-cube resume."""
        import pickle
        from dataclasses import replace
        from .types import Placement
        from box_extract.planning import flush_local
        from so101_pick_place.experiments.box_extraction.planner import Phase
        saved = self.pending_grasp
        if saved is None or saved['calibration_sha256']!=self.geometry.fingerprint:
            raise ValueError('no matching supported-grasp checkpoint')
        with (self.output/'last-relocate-plan.pkl').open('rb') as handle:
            plan = pickle.load(handle)
        target = saved['target_id']
        evidence = next(p.placed_cube.evidence for p in reversed(plan.phases) if p.placed_cube is not None)
        quad = np.asarray(evidence['quad'])
        frame = self.cameras.get_latest_frame('overview')
        mask = np.zeros(frame.image.shape[:2],np.uint8);cv2.fillConvexPoly(mask,np.int32(quad),255)
        obj = self.geometry.estimate_cube_pose({'quad':quad,'mask':mask,'color':saved['color'],'confidence':.98},frame,evidence['support_z_m'])
        obj.id,obj.timestamp = target,float(evidence.get('spatial_timestamp',saved['feedback_time']))
        obj.evidence.update(evidence,metric_qualified=True)
        with self.scene_lock:
            self.scene.state.objects[target] = obj
            self.scene.track_memory[target] = obj
            self.resume_original = self.scene.native_object(target)
        pose = obj.pose.copy();pose[:3,3] = plan.placement_center_m
        placement = Placement(pose,'box-1')
        native = self.scene.native_scene(target,placement)
        self.motion.qualification_blockers(self.geometry)
        world = self.motion.make_world(native)
        self.ensure_driver()
        state = self.robot.read_state();current=state['positions']
        if max(abs(current[j]-saved['positions'][j]) for j in current if j!='gripper')>1.5 or abs(current['gripper']-saved['positions']['gripper'])>2.:
            raise ValueError('supported grasp changed since the force-confirmed checkpoint')
        # A lift may have been commanded but refused at the next phase boundary.
        # Regrasp is permissible only while the actual attached cube is still on
        # the original support, never merely because the old phase was lift_off.
        previous_frame = self.robot.backend.model.frames(saved['positions'])['gripper_frame_link']
        previous_local = np.asarray(saved.get('payload_local',flush_local(
            (self.resume_original.vertices_m-previous_frame[:3,3])@previous_frame[:3,:3])))
        supported = self.robot.backend._transform_points(
            self.robot.backend.model.frames(current)['gripper_frame_link'],previous_local)
        support_error = float(supported[:,2].min()-evidence['support_z_m'])
        if abs(support_error)>.001:
            raise ValueError(f'supported regrasp refused: actual cube bottom {support_error:.4f} m from original support')
        self.metrics['supported_resume_bottom_error_m'] = support_error
        driver = self.motion.driver
        driver.phase,driver.plan,driver.world = Phase('grasp',(current,current),jaw_contact=True),plan,world
        if self.scene.held is None or not driver.retention.armed:
            self.robot.enable()
            driver.force_close_active = True
            try:
                self.robot.arm.grasp(rate=6.,past=2.,squeeze=1.,empty=4.,step='resume_grasp')
            finally:
                driver.force_close_active = False
        actual,_,_ = self.robot.arm.read()
        if actual['gripper']<5. or abs(actual['gripper']-saved['positions']['gripper'])>4.:
            raise RuntimeError('supported regrasp differs from the confirmed cube width')
        for attempt in range(5):
            self.check_cancel();self.cameras.check()
            try:
                local = flush_local(driver.confirm_grasp())
                break
            except RuntimeError as exc:
                if 'grasp contact force is not confirmed' not in str(exc):
                    raise
                time.sleep(.04)
        else:
            raise RuntimeError('grasp contact force is not confirmed; transport is blocked')
        driver.payload_local_override = local
        actual = self.robot.read_state()['positions']
        transform = np.linalg.inv(self.robot.backend.model.frames(actual)['gripper_frame_link'])@obj.pose
        transform[:3,3] = local.mean(0)
        self.scene.attach_object(target,transform,obj)
        saved.update(positions=actual,feedback_time=time.time(),payload_local=local.tolist())
        self.pending_grasp_file.write_text(json.dumps(saved,indent=2))
        trial = json.loads((self.output/'contact-free-trial.json').read_text())
        if not trial.get('passed') or trial['attachment_sha256']!=self.motion.attachment.fingerprint:
            raise ValueError('matching physical approach evidence is missing')
        self.motion.contact_free_passed = True
        self.motion.trial_attachment_sha256 = self.motion.attachment.fingerprint
        self.resume_plan = plan
        self.mode = 'relocate'
        failures = []
        for patch in self.scene.find_free_box_slots('box-1',obj.dimensions):
            center = np.r_[np.mean(patch.polygon_xy_m,axis=0),patch.height_m+obj.dimensions[2]/2]
            candidate_pose = obj.pose.copy();candidate_pose[:3,3] = center
            candidate = Placement(candidate_pose,'box-1')
            native = self.scene.native_scene(target,candidate)
            native = replace(native,cubes=native.cubes|{target:self.resume_original})
            candidate_world = self.motion.make_world(native)
            attached_frame = self.robot.backend.model.frames(actual)['gripper_frame_link']
            vertices = self.robot.backend._transform_points(attached_frame,local)
            cube = replace(self.resume_original,center_m=vertices.mean(0),vertices_m=vertices)
            policy = {'target':target,'jaw_contact':True,'payload_local':local,'support_contact':True}
            rejections = []
            def acceptable(q):
                for name,pose in (('pose',q),('command',self.motion.planner._command_pose(q))):
                    reason = candidate_world.reason(pose,**policy)
                    if reason:
                        rejections.append(name+': '+reason)
                        return False
                return True
            try:
                self.motion.planner.placement_pose(actual,local,cube,center+[0.,0.,.001],accept_pose=acceptable)
                previous = replace(plan,payload_local=local)
                remainder,updated = self.motion.replan_held(replace(native,cubes={k:v for k,v in native.cubes.items() if k!=target}),actual,previous,self.resume_original)
                checked_world = self.motion.make_world(updated)
                self.motion.check_path(remainder,checked_world)
                self.resume_remainder = remainder,updated,checked_world
                placement = candidate
                break
            except ValueError as exc:
                failures.append(str(exc)+'; '+'; '.join(rejections[-1:]))
                self.metrics['placement_rejections'] = failures[-6:]
        else:
            raise ValueError('held placement IK/compensated clearance: '+'; '.join(failures[-3:]))
        return self.actions.relocate(target,placement,resume_held=True)

    def resume_released_placement(self, attendance):
        """Finish only the empty-jaw retreat and verification after release."""
        from dataclasses import replace
        from .types import ActionResult
        from so101_pick_place.experiments.box_extraction.planner import Phase
        saved = self.read_release_checkpoint()
        if saved['calibration_sha256']!=self.geometry.fingerprint or self.scene.held is not None:
            raise ValueError('released-placement recovery has no matching empty-hand checkpoint')
        state = self.robot.read_state()
        actual = state['positions']
        if (abs(state['loads']['gripper']) or abs(state['currents']['gripper']) or
            state['gripper']<saved['release_actual']['gripper']-self.robot.encoder_steps()['gripper']):
            raise ValueError('retreat: empty jaws are not confirmed by native feedback')
        self.geometry.grasp_mode = self.motion.grasp_mode = 'overhead_grasp'
        self.motion.operator_observation = attendance
        self.motion.qualification_blockers(self.geometry)
        self.scene.state.boxes = saved['boxes']
        self.scene.metric_boxes = saved['boxes'].copy()
        self.scene.boxes_semantically_seeded = True
        self.scene.state.objects[saved['object'].id] = saved['object']
        self.scene.track_memory[saved['object'].id] = saved['object']
        self.scene.metric_objects.update(saved['native_scene'].cubes)
        self.scene.pending_placement = saved['object'].id,saved['object'].color
        self.refresh()
        world = self.motion.make_world(saved['native_scene'])
        original = saved['plan']
        placed = next(p.placed_cube for p in reversed(original.phases) if p.placed_cube is not None)
        refusal = world.reason(actual,target=original.target_id,jaw_contact=True,placed_cube=placed)
        if refusal:
            raise ValueError('retreat actual start: '+refusal+'; calculate RECOVERY_FROM_INVALID_START while retaining STOP')
        trial = json.loads((self.output/'contact-free-trial.json').read_text())
        if not trial.get('passed') or trial['attachment_sha256']!=self.motion.attachment.fingerprint:
            raise ValueError('matching physical approach evidence is missing')
        self.motion.contact_free_passed = True
        self.motion.trial_attachment_sha256 = self.motion.attachment.fingerprint
        world.release_origin = actual
        path = self.motion.planner._retract_path(actual,original.payload_local.mean(0),world,
            target=original.target_id,jaw_contact=True,placed_cube=placed)
        phase = Phase('retract',path,jaw_contact=True,placed_cube=placed)
        plan = replace(original,phases=(phase,))
        self.motion.check_path(plan,world)
        self.motion.plan_park(phase.joints[-1],plan,world)
        self.ensure_driver()
        self.motion.contact_free_passed = True
        self.motion.trial_attachment_sha256 = self.motion.attachment.fingerprint
        self.robot.enable()
        self.motion.execute_trajectory(phase,plan,world)
        verified,pose = self.actions.verify_placement(saved['object'],saved['placement'])
        if not verified:
            raise ValueError('released cube is not observed within placement tolerance')
        self.scene.detach_object(original.target_id,pose)
        observed_scene = replace(saved['native_scene'],cubes=saved['native_scene'].cubes|{original.target_id:self.scene.native_object(original.target_id)})
        self.motion.plan,self.motion.world = plan,self.motion.make_world(observed_scene)
        parked = False
        if self.park_qualified_file.exists():
            self.park_arm(plan,self.motion.world)
            parked = True
        self.physical_relocate_count += 1
        self.report('completed','Зелёный кубик в белой коробке; retreat завершён; '+('PARKED' if parked else 'Park ожидает первого испытания под наблюдением'))
        return ActionResult(True,'released placement observed; empty-jaw retreat completed',original.target_id,self.scene.state,
                            {'physical_verified':True,'retreat_completed':True,'parked':parked,'motion_started':True})

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
            reasons = self.blockers() if not dry else []
            if reasons:
                raise ValueError('; '.join(reasons))
            self.robot.first_command_at = None
            self.started = time.monotonic()
            self.finished = None
            self.mode = mode
            self.scene.remember_reset_layout()
            self.running = True
            self.metrics = {key:None for key in ('first_command_s','vision_s','grasp_planning_s','ik_s',
                'collision_checking_s','motion_planning_s','actual_motion_s','verification_s','cycle_s')}
            self.metrics['timing_scope'] = 'inclusive stages; IK includes native IK geometry, with collision calls subtracted in vertical IK'
            self.motion.metrics = self.metrics
            self.motion.planner.timings = self.metrics
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

    def refine_object(self, object_id, plan, world):
        """Bounded active perception inside a previously checked working scene."""
        from dataclasses import replace
        from so101_pick_place.experiments.box_extraction.planner import Phase
        self.check_cancel()
        force_held=self.motion.driver is not None and self.motion.driver.retention.armed
        if self.scene.held is not None or force_held or not self.robot.audit_state(self.robot.read_state())['planning_valid']:
            raise ValueError('active perception is allowed only with an empty jaw in the verified working area')
        obj = self.scene.state.objects[object_id]
        point = obj.pose[:3,3]+[0.,0.,obj.dimensions[2]/2]
        current = self.robot.read_state()['positions']
        candidates = self.motion.informative_views(plan,world,self.geometry,point,current,float(np.linalg.norm(obj.dimensions)/2))
        attempted = []
        for goal in candidates[:3]:
            self.check_cancel();self.refresh()
            if not self.scene.state.evidence.get('coverage_verified'):
                raise ValueError('scene coverage changed; an unobserved region is not assumed empty')
            for key,cube in world.scene.cubes.items():
                observed = self.scene.state.objects.get(key)
                if observed is None or not observed.evidence.get('metric_qualified'):
                    raise ValueError('active-view path lost its measured obstacle map')
                sigma = max(observed.evidence.get('position_sigma_m',[.003]*3))
                if np.linalg.norm(observed.pose[:3,3]-cube.center_m)>3*sigma:
                    raise ValueError('workspace changed before active observation')
            current = self.robot.read_state()['positions']
            self.report('observing','информативный ракурс из проверенного подхода')
            path,_ = self.motion.planner.transport(current,dict(goal,gripper=current['gripper']),world,target=object_id)
            phase = Phase('observe',path)
            self.motion.check_path(replace(plan,phases=(phase,)),world)
            self.motion.execute_trajectory(phase,plan,world)
            after_motion = time.time()
            before = self.robot.read_state();self.cameras.record_state(before['timestamp'],before['q'],before['gripper'])
            for camera in ('overview','wrist'):
                self.cameras.wait_for_frame(camera,after_motion,self.check_cancel)
            after = self.robot.read_state();self.cameras.record_state(after['timestamp'],after['q'],after['gripper'])
            self.refresh()
            measured = self.scene.state.objects[object_id]
            attempted.append({'q_rad':after['q'].tolist(),'frame_id':self.cameras.get_latest_frame('wrist').frame_id})
            self.metrics['active_observations'] = attempted
            if measured.evidence.get('metric_qualified') and max(measured.evidence['position_sigma_m'])<=.003:
                return measured
        raise ValueError('three checked observations did not reach grasp accuracy; holding without blind retry')

    def record_action_measurement(self, event, object_id, values):
        """Observe residuals; never apply an automatic offset or change servo limits."""
        state = self.robot.read_state()
        record = {'event':event,'object_id':object_id,'timestamp':time.time(),'q_rad':state['q'].tolist(),
            'gripper_native':state['gripper'],'calibration_sha256':self.geometry.fingerprint,
            'scene_version':self.scene.state.version,'values':values}
        self.action_measurements.append(record)
        try:
            with (self.output/'action-measurements.jsonl').open('a') as handle:
                handle.write(json.dumps(record,ensure_ascii=False)+'\n')
        except OSError as exc:
            self.metrics['measurement_log_error'] = str(exc)
        independent = []
        for item in self.action_measurements:
            if item['event']=='placement' and not any(item['object_id']==old['object_id'] and np.max(np.abs(np.subtract(item['q_rad'],old['q_rad'])))<np.deg2rad(5.) for old in independent):
                independent.append(item)
        if independent:
            errors = np.asarray([item['values']['error_m'] for item in independent])
            median = np.median(errors,axis=0);spread=1.4826*np.median(np.abs(errors-median),axis=0)
            sigma = max(max(item['values']['position_sigma_m']) for item in independent)
            significant = len(independent)>=5 and bool(np.any(np.abs(median)>2*np.maximum(spread/np.sqrt(len(independent)),sigma)))
            self.metrics['placement_residuals'] = {'independent_samples':len(independent),'median_error_m':median.tolist(),
                'robust_spread_m':spread.tolist(),'systematic_bias_candidate':significant,
                'parameters_changed':False,'hardware_limits_changed':False}

    def execute(self, payload):
        try:
            self.check_cancel()
            if self.robot.arm is None:
                self.robot.read_state()
            self.refresh()
            selected = payload.get('object_id') or self.selected
            destination = payload.get('destination') or self.destination
            if payload.get('auto_select') and payload['mode']=='relocate':
                from .tasks import placement_in_box
                self.ensure_driver()
                self.results = []
                for row in self.actions.rank_first_pick(destination,payload.get('exclude_colors',())):
                    if not self.results or not (self.results[-1].success or self.results[-1].evidence.get('motion_started') or self.results[-1].evidence.get('critical')):
                        self.refresh()
                        result = self.actions.relocate(row['object_id'],placement_in_box(self.scene,row['object_id'],destination),dry=payload['dry'])
                        self.results.append(result)
                if not self.results:
                    raise ValueError('none of the observed cubes has an admissible initial grasp')
            elif payload['mode'] in self.EXTRA_MODES and not payload['dry']:
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
            if not payload['dry'] and not self.stop_event.is_set() and not self.parked and self.park_qualified_file.exists():
                try:
                    self.park_arm(self.motion.plan,self.motion.world)
                except ValueError as exc:
                    self.park_error = str(exc)
            self.report('completed' if ok else 'blocked' if self.stop_event.is_set() else 'recovering' if self.scene.held is not None else 'failed',
                'Все запрошенные переносы подтверждены' if ok else self.results[-1].reason)
        except Exception as exc:
            self.report('blocked' if self.stop_event.is_set() or self.blockers() else 'failed',f'{type(exc).__name__}: {exc}')
            self.results = [{'success':False,'reason':str(exc),'physical_verified':False}]
            if not self.stop_event.is_set() and self.scene.held is None and self.park_qualified_file.exists():
                try:
                    self.park_arm(self.motion.plan,self.motion.world)
                except ValueError as park_error:
                    self.park_error = str(park_error)
        finally:
            self.finished = time.monotonic()
            if self.last_plan is not None:
                import pickle
                with (self.output/'last-relocate-plan.pkl').open('wb') as handle:
                    pickle.dump(self.last_plan,handle)
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
        conditions = self.blocking_conditions()
        reasons = [item['reason'] for item in conditions]
        audit = None if self.robot.last_state is None else self.robot.audit_state(self.robot.last_state)
        if not self.initialized:
            state = 'NOT_INITIALIZED'
        elif self.stop_event.is_set():
            state = 'SAFETY_BLOCKED'
        elif self.parked:
            state = 'PARKED'
        elif not self.running and self.scene.held is not None:
            state = 'RECOVERING'
        elif self.running:
            state = 'EXECUTING' if self.phase in ('moving','contact_free') else 'RECOVERING' if self.phase in ('recovering','recovered') else 'PLANNING'
        elif reasons:
            state = 'SAFETY_BLOCKED'
        else:
            state = {'completed':'COMPLETED','failed':'FAILED'}.get(self.phase,'READY')
        physical_blocker = None
        if audit is not None and not audit['planning_valid']:
            physical_blocker = {'code':'CURRENT_POSE_CLEARANCE','automatic_exit_authorized':False,
                'clearance_m':audit['minimum_model_clearance_m'],'required_clearance_m':audit['required_clearance_m'],
                'classification':'SOFTWARE_MARGIN' if not audit['hardware_limit_problems'] and not audit['zero_gap_sample_contact'] else 'GEOMETRY_OR_LIMIT',
                'physical_clearance_measured':False,
                'required_action':'Рассчитать retreat из фактической позы с неподвижным уложенным кубиком; отдельный Park возможен только после выхода в рабочие зазоры.',
                'recheck':'Свежие энкодеры и штатная проверка всего пути; сброс STOP сам по себе движение не разрешает.'}
            if self.recovery_proposal:
                physical_blocker['required_action'] = self.recovery_proposal.get('reason',physical_blocker['required_action'])
        return {'running':self.running,'mode':self.mode,'dry':None,'started':self.started,'finished':self.finished,
            'phase':self.phase,'state':state,'blocking_conditions':conditions,
            'project_detail':self.detail,'project_state':state,'detail':self.detail,
            'ready':self.initialized and not self.running and not reasons,'blockers':reasons,'paused':self.pause_event.is_set(),
            'stop_latched':self.stop_event.is_set(),'hardware_fault_latched':self.fault_latched,'log':list(self.log),'exit_code':0 if self.phase=='completed' else None,
            'metrics':self.metrics,'selected':self.selected,'scene_version':self.scene.state.version,
            'motor_health':self.robot.motor_health,'motor_warnings':self.motor_warnings,
            'motor_loads':{} if self.robot.last_state is None else self.robot.last_state['loads'],
            'hold_duration_s':max(0.,time.time()-self.last_motion_unix),'parked':self.parked,
            'park_error':self.park_error,'recovery_proposal':self.recovery_proposal,'physical_progress':self.physical_progress,
            'robot_audit':audit,'physical_blocker':physical_blocker,
            'known_working_configuration':getattr(self,'known_working_configuration',None),
            'startup_driver':None if self.robot.arm is None else type(self.robot.arm).__name__,
            'startup_preparation_available':self.robot.arm is not None and hasattr(self.robot.arm,'prepare_ready'),
            'grasp_mode':self.motion.grasp_mode,'wrist_refinement_enabled':self.motion.grasp_mode=='wrist_refinement',
            'supervision_required':self.motion.grasp_mode=='overhead_grasp',
            'physical_relocate_verified':any(not isinstance(r,dict) and r.success and r.evidence.get('physical_verified') for r in self.results),'experiments':{m:'blocked_pending_relocate_qualification' for m in self.MODES},
            'results':[r if isinstance(r,dict) else {'success':r.success,'reason':r.reason,'evidence':r.evidence} for r in self.results]}

    def scene_public(self):
        with self.scene_lock:
            data = self.scene.public()
        now = time.time()
        data['camera_status'] = {name:{'frame_id':frame.frame_id,'capture_interval':list(frame.capture_interval),
            'captured_at':frame.timestamp,'received_at':frame.received_at,'age_s':now-frame.capture_interval[0],
            'fresh':now-frame.capture_interval[0]<=self.cameras.max_age} for name,frame in dict(self.cameras.frames).items()}
        data['video_fresh'] = len(data['camera_status'])==2 and all(value['fresh'] for value in data['camera_status'].values())
        data['detection_fresh'] = self.scene_error is None and self.last_refresh is not None and now-self.last_refresh<=2.
        data['observation_valid'] = data['video_fresh'] and data['detection_fresh'] and self.camera_error is None
        data['observation_error'] = self.camera_error or self.scene_error
        data['detection_error'] = self.scene_error
        data['camera_cache_refreshes'] = list(self.cameras.cache_refreshes)
        return data

    def persist(self):
        target = self.output/'status.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.status(),ensure_ascii=False,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)))
        temporary.replace(target)

    def loop(self):
        self.initialize()
        last_persist = 0.
        while True:
            try:
                payload = self.commands.get(timeout=.04)
            except queue.Empty:
                try:
                    if self.robot.arm is not None:
                        state = self.robot.read_state()
                        self.cameras.record_state(state['timestamp'],state['q'],state['gripper'])
                        self.monitor_motors()
                        if self.scene.held is not None and self.motion.driver is not None:
                            self.motion.driver.retention.check(state['loads']['gripper'],state['currents']['gripper'])
                    if not self.cameras_initialized and time.monotonic()>=self.camera_retry_at:
                        self.initialize_cameras()
                        if self.cameras_initialized:
                            reasons = self.blockers()
                            self.report('blocked' if reasons else 'ready','; '.join(reasons) if reasons else 'Ready')
                    native = self.cameras.json('/api/status')
                    state = native.get('state')
                    if state=='EMERGENCY_STOP':
                        self.native_stop(native.get('updated_at'))
                    self.last_native_state = state
                    if (self.park_qualified_file.exists() and not self.stop_event.is_set() and not self.parked and
                        not self.running and self.scene.held is None and
                        self.idle_park_attempted_at<self.last_motion_unix and
                        time.time()-self.last_motion_unix>=self.config['idle_park_after_s']):
                        self.idle_park_attempted_at = time.time()
                        self.running = True
                        try:
                            self.park_arm()
                        except ValueError as exc:
                            self.park_error = str(exc)
                        finally:
                            self.running = False
                except Exception as exc:
                    self.hardware_error = str(exc)
                    if self.robot.arm is not None and self.robot.arm.torque and not self.stop_event.is_set():
                        self.stop(reason='feedback/watchdog: '+str(exc),hardware_fault=True)
                if time.monotonic()-last_persist > 2.:
                    self.persist()
                    last_persist = time.monotonic()
            else:
                if payload.get('control')=='prepare':
                    self.perform_prepare(payload)
                elif payload.get('control')=='reset':
                    try:
                        self.perform_reset()
                    except Exception as exc:
                        self.stop(reason='reset refused: '+str(exc),forward=False)
                elif payload.get('control')=='park':
                    self.perform_park(payload)
                else:
                    self.execute(payload)
