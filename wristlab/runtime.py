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
        self.scene_error = None
        self.scene_lock = self.scene.lock
        self.preparing_scene = False
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
        try:
            self.metrics['camera_controls'] = self.cameras.apply_controls()
            self.cameras.start_capture()
            self.cameras.on_native_stop = self.native_stop
            self.cameras.on_frames = self.background_refresh
            # The line-only detector confused table/cable edges with boxes in
            # this scene. Reuse the already-loaded native model once, before
            # Ready; experiments only track the resulting current observation.
            self.scene.boxes_semantically_seeded = True
            box_started = time.perf_counter()
            try:
                scan = self.cameras.json('/api/scan-hybrid',{},timeout=30.)
                self.scene.install_box_proposals(scan)
                (self.output/'box-observation.json').write_text(json.dumps(scan))
            finally:
                self.metrics['box_detection_startup_s'] = time.perf_counter()-box_started
            wrist_path = self.output/'wrist-camera-calibration.json'
            if wrist_path.exists():
                self.geometry.install_wrist_calibration(json.loads(wrist_path.read_text()),self.robot)
            self.refresh(measure_attachment=True)
        except Exception as exc:
            self.report('blocked',str(exc))
        self.initialized = True
        reasons = self.blockers()
        if not reasons:
            self.ensure_driver()
        self.report('blocked' if reasons else 'ready','; '.join(reasons) if reasons else 'Ready')
        self.persist()

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
        if self.scene.held is not None and not self.running:
            add('PAYLOAD_HELD','RECOVERY','после ошибки планирования предмет удерживается',
                'предмет установлен по отдельно проверенному плану восстановления; автоматическое раскрытие запрещено')
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
        self.check_cancel()
        self.cameras.check()
        self.motion.guarded_tick(record)
        stamp = time.time()
        self.robot.last_state = {'timestamp':stamp,'positions':dict(record['pos']),
            'loads':dict(record['load']),'currents':dict(record['ma']),
            'q':self.robot.from_native(record['pos']),'gripper':record['pos']['gripper']}
        self.cameras.record_state(stamp,self.robot.from_native(record['pos']),record['pos']['gripper'])
        if self.robot.first_command_at is None:
            self.robot.first_command_at = time.monotonic()

    def prepare(self, payload):
        with self.submit_lock:
            if self.running or not self.commands.empty():
                raise ValueError('another task is running or queued')
            self.check_cancel()
            if not self.initialized or self.scene.held is not None:
                raise ValueError('startup requires initialized feedback and an empty hand')
            if payload.get('cable_and_hidden_surfaces_clear') is not True:
                raise ValueError('operator observation is required for this attended startup')
            self.running = True
            self.commands.put_nowait(dict(payload,control='prepare'))

    def perform_prepare(self, payload):
        self.preparing_scene = True
        try:
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
                recovery = bool(payload.get('recover_empty_approach'))
                result = self.prepare_overhead(attendance,recovery=recovery)
                if recovery:
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
            if self.robot.first_command_at is not None and (payload.get('grasp_mode')!='overhead_grasp' or payload.get('recover_empty_approach')):
                self.stop(reason='startup: '+str(exc),hardware_fault=True)
        finally:
            if self.robot.first_command_at is not None:
                self.metrics['first_command_s'] = self.robot.first_command_at-self.started
            self.running = False
            self.preparing_scene = False
            self.persist()

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
            wrist_colors = {d['color'] for d in detect_cubes(self.scene.frames['wrist'])}
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
        sources = [json.loads((self.output/name).read_text()) for name in
                   (('pre-approach-overview-reference.json','overview-confirm-0.json','overview-confirm-1.json') if recovery else
                    ('box-observation.json','overview-confirm-0.json','overview-confirm-1.json'))]
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
        return {'grasp_mode':'overhead_grasp','wrist_refinement':False,'overview_control':control,
                'sources':[s['artifacts'] for s in sources],'assumptions':params.__dict__,
                'operator_observation':attendance,'physical_pick_place_completed':False}

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
            self.report('completed' if ok else 'blocked' if self.stop_event.is_set() else 'recovering' if self.scene.held is not None else 'failed',
                'Все запрошенные переносы подтверждены' if ok else self.results[-1].reason)
        except Exception as exc:
            self.report('blocked' if self.stop_event.is_set() or self.blockers() else 'failed',f'{type(exc).__name__}: {exc}')
            self.results = [{'success':False,'reason':str(exc),'physical_verified':False}]
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
            physical_blocker = {'code':'UNQUALIFIED_FOLDED_START','automatic_exit_authorized':False,
                'clearance_m':audit['minimum_model_clearance_m'],'required_clearance_m':audit['required_clearance_m'],
                'classification':'SOFTWARE_MARGIN' if not audit['hardware_limit_problems'] and not audit['zero_gap_sample_contact'] else 'GEOMETRY_OR_LIMIT',
                'physical_clearance_measured':False,
                'missing_measurement':'Минимальный зазор камеры/крепления/кабеля до руки на начальном участке выхода из текущей позы с учётом отставания приводов; wrist_flex сейчас 62,59°, в последнем записанном camera-aware старте было 15,38°',
                'required_action':'Калибровка руки сохранена. Восстановлен MountedTaskArm.prepare_ready; его вход из сегодняшней позы требует проверки зазора на дополнительном начальном участке. Слепой native-unfold включать нельзя.',
                'recheck':'Свежие энкодеры и штатная проверка всего пути; сброс STOP сам по себе движение не разрешает.'}
            observation=getattr(self.scene,'passive_measurements',None)
            if observation and observation.get('unresolved_part'):
                physical_blocker['unobserved_part']=observation['unresolved_part']
                physical_blocker['image_region_px']=observation['unresolved_region_px']
                physical_blocker['missing_measurement']=observation['needed_measurement']
                physical_blocker['visible_surface_bound']=observation.get('visible_surface_clearance_bound')
        return {'running':self.running,'mode':self.mode,'dry':None,'started':self.started,'finished':self.finished,
            'phase':self.phase,'state':state,'blocking_conditions':conditions,
            'project_detail':self.detail,'project_state':state,'detail':self.detail,
            'ready':self.initialized and not self.running and not reasons,'blockers':reasons,'paused':self.pause_event.is_set(),
            'stop_latched':self.stop_event.is_set(),'hardware_fault_latched':self.fault_latched,'log':list(self.log),'exit_code':0 if self.phase=='completed' else None,
            'metrics':self.metrics,'selected':self.selected,'scene_version':self.scene.state.version,
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
                        if self.scene.held is not None and self.motion.driver is not None:
                            self.motion.driver.retention.check(state['loads']['gripper'],state['currents']['gripper'])
                    if self.cameras.on_frames is None and time.monotonic()-last_vision >= self.config['poll_interval_s']:
                        self.refresh()
                        last_vision = time.monotonic()
                    native = self.cameras.json('/api/status')
                    state = native.get('state')
                    if state=='EMERGENCY_STOP':
                        self.native_stop(native.get('updated_at'))
                    self.last_native_state = state
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
                else:
                    self.execute(payload)
