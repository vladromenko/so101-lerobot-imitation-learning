"""The universal relocate operation; all experiments share this physical path."""
from dataclasses import replace
import time
import numpy as np
from so101_pick_place.experiments.box_extraction.planner import Phase
from .grasp import choose_grasp, visual_alignment
from .types import ActionResult


class Actions:
    def __init__(self, runtime):
        self.runtime = runtime
        self.scene,self.robot,self.motion = runtime.scene,runtime.robot,runtime.motion

    def observe_object(self, object_id):
        self.runtime.refresh()
        obj = self.scene.state.objects[object_id]
        if time.time()-obj.timestamp > .5:
            raise ValueError('fresh target observation is required')
        return obj

    def relocate(self, object_id, placement, dry=False, recovery_attempt=0):
        rt = self.runtime
        started, motion_started = time.monotonic(), False
        executing,last_phase = False,None
        stage = 'observe'
        plan = world = None
        aligned,active_refined = False,False
        try:
            obj = self.observe_object(object_id)
            self.scene.reserve_object(object_id)
            rt.report(stage,object_id)
            reasons = self.motion.qualification_reasons(self.scene.geometry)
            if reasons:
                raise ValueError('; '.join(reasons))
            stage = 'grasp_planning'
            rt.report(stage,object_id)
            native_scene = self.scene.native_scene(object_id,placement)
            if not dry:
                rt.ensure_driver()
            start = self.robot.read_state()['positions']
            world = self.motion.make_world(native_scene)
            candidate,plan,world = choose_grasp(obj,native_scene,self.robot,self.motion.planner,start,rt.check_cancel,world)
            stage = 'path_check'
            self.motion.check_path(plan,world)
            rt.last_plan = plan
            if dry:
                return ActionResult(True,'geometric path checked; no actuator commands',object_id,self.scene.state,
                                    {'physical_verified':False,'motion_started':False})
            if not self.motion.contact_free_passed:
                motion_started = True
                executing = True
                self.motion.contact_free_trial(plan,world,start)
                executing = False
            self.motion.plan,self.motion.world = plan,world
            held_verified = False
            wrist_anchor = None
            queue = list(plan.phases)
            while queue:
                rt.wait_pause()
                phase = queue.pop(0)
                stage = phase.name
                if stage == 'enter' and not aligned:
                    stage = 'visual_alignment'
                    try:
                        observation = visual_alignment(obj,rt.cameras,self.robot,self.motion,plan,world,self.scene.geometry)
                    except ValueError:
                        if active_refined or not hasattr(rt,'refine_object'):
                            raise
                        obj = rt.refine_object(object_id,plan,world)
                        active_refined,observation = True,None
                    if observation is not None:
                        observed_top = np.asarray(observation['observed_top_m'])
                        delta = observed_top-(obj.pose[:3,3]+[0.,0.,obj.dimensions[2]/2])
                        if hasattr(rt,'record_action_measurement'):
                            rt.record_action_measurement('visual_alignment',object_id,{'target_top_m':(obj.pose[:3,3]+[0.,0.,obj.dimensions[2]/2]).tolist(),
                                'observed_top_m':observed_top.tolist(),'xy_error_m':delta[:2].tolist(),'frame_id':observation['frame_id'],
                                'independent_z_measurement':False})
                        maximum = 3*max(np.asarray(obj.evidence.get('position_sigma_m',[.003]*3))[:2])
                        if np.linalg.norm(delta[:2])>maximum:
                            raise ValueError('wrist correction exceeds measured localization uncertainty')
                        original = native_scene.cubes[object_id]
                        updated = replace(original,center_m=original.center_m+delta,vertices_m=original.vertices_m+delta)
                        self.scene.metric_objects[object_id] = updated
                    rt.refresh()
                    obj = self.scene.state.objects[object_id]
                    native_scene = self.scene.native_scene(object_id,placement)
                    current = self.robot.read_state()['positions']
                    world = self.motion.make_world(native_scene)
                    stage = 'grasp_planning'
                    candidate,plan,world = choose_grasp(obj,native_scene,self.robot,self.motion.planner,current,rt.check_cancel,world)
                    stage = 'path_check'
                    self.motion.check_path(plan,world)
                    queue = list(plan.phases)
                    aligned = observation is not None
                    phase = None
                if phase is not None:
                    if stage == 'release' and not held_verified:
                        raise RuntimeError('release refused: independent holding evidence is missing')
                    motion_started = True
                    executing = True
                    self.motion.execute_trajectory(phase,plan,world)
                    executing,last_phase = False,phase
                    if stage == 'grasp':
                        from box_extract.planning import flush_local
                        local = flush_local(self.motion.driver.confirm_grasp())
                        self.motion.driver.payload_local_override = local
                        actual = self.robot.arm.read()[0]
                        transform = np.linalg.inv(self.robot.backend.model.frames(actual)['gripper_frame_link']) @ obj.pose
                        transform[:3,3] = local.mean(0)
                        self.scene.attach_object(object_id,transform)
                        plan = replace(plan,payload_local=local)
                        wrist_anchor = self.capture_grip_view(obj,time.time())
                        # Post-contact attachment is derived from measured encoders.
                        adjusted = plan
                        stage = 'held_planning'
                        planning_started = time.perf_counter()
                        try:
                            remainder,new_scene = self.motion.replan_held(replace(native_scene,cubes={key:value for key,value in native_scene.cubes.items() if key!=object_id}),actual,adjusted,native_scene.cubes[object_id])
                        finally:
                            rt.metrics['motion_planning_s'] = (rt.metrics.get('motion_planning_s') or 0.)+time.perf_counter()-planning_started
                        world = self.motion.make_world(new_scene)
                        stage = 'path_check'
                        self.motion.check_path(remainder,world)
                        plan = remainder
                        queue = list(remainder.phases)
                    elif stage == 'extract_vertical':
                        verify_started = time.perf_counter()
                        held_verified = self.verify_pick(obj,wrist_anchor,time.time())
                        rt.metrics['verification_s'] = (rt.metrics.get('verification_s') or 0.)+time.perf_counter()-verify_started
                        if not held_verified:
                            raise RuntimeError('holding not confirmed by jaw feedback and wrist observation')
                    elif stage == 'release':
                        self.motion.driver.release_confirmed()
                        self.scene.held = None
                        self.scene.pending_placement = (object_id,obj.color)
            stage = 'verify_placement'
            verify_started = time.monotonic()
            verified,observed = self.verify_placement(obj,placement)
            rt.metrics['verification_s'] = (rt.metrics.get('verification_s',0.) or 0.)+time.monotonic()-verify_started
            if not verified and recovery_attempt < rt.recovery.max_attempts:
                candidates = [o for o in self.scene.state.objects.values() if o.color==obj.color and o.evidence.get('metric_qualified')]
                if len(candidates)==1:
                    self.scene.release_reservation(object_id)
                    return self.relocate(candidates[0].id,placement,recovery_attempt=recovery_attempt+1)
            if not verified:
                raise ValueError('object was not observed at placement within tolerance')
            self.scene.detach_object(object_id,observed)
            if hasattr(rt,'record_action_measurement'):
                measured = self.scene.state.objects[object_id]
                rt.record_action_measurement('placement',object_id,{'requested_pose_m':placement.object_pose.tolist(),
                    'observed_pose_m':observed.tolist(),'error_m':(observed[:3,3]-placement.object_pose[:3,3]).tolist(),
                    'position_sigma_m':measured.evidence['position_sigma_m']})
            returned = rt.physical_relocate_count==0 or rt.mode=='relocate'
            if returned:
                stage = 'return_planning'
                self.motion.return_to_start(start,plan,world,self.scene.native_object(object_id))
            rt.physical_relocate_count += 1
            rt.metrics['cycle_s'] = time.monotonic()-started
            return ActionResult(True,'relocated and observed at destination',object_id,self.scene.state,
                {'physical_verified':True,'motion_started':True,'placement_tolerance_m':placement.tolerance,'returned_to_start':returned})
        except Exception as exc:
            reason = f'{stage}: {type(exc).__name__}: {exc}'
            empty = stage=='grasp' and str(exc)=='measured grasp differs from jaw-mesh prediction'
            if empty and plan is not None and world is not None:
                try:
                    actual = self.robot.read_state()
                    if actual['gripper']<4. and not self.motion.driver.retention.armed:
                        self.recover_empty_grasp(object_id,placement,plan)
                        return ActionResult(False,reason,object_id,self.scene.state,
                            {'motion_started':True,'recovered':True,'critical':False,'physical_verified':False})
                except Exception as recovery_error:
                    reason += '; recovery refused: '+str(recovery_error)
            cancelled = rt.stop_event.is_set()
            routine = isinstance(exc,(ValueError,KeyError)) and not executing
            holding = self.scene.held is not None
            if routine and motion_started and not cancelled:
                try:
                    rt.cameras.check()
                    actual = self.robot.read_state()
                    if holding:
                        self.motion.driver.confirm_retention()
                    if last_phase is not None:
                        policy_phase = replace(last_phase,carrying=holding,jaw_contact=holding or last_phase.jaw_contact,
                                               support_contact=last_phase.support_contact or (holding and last_phase.name=='grasp'))
                        safety = self.motion.state_is_valid(actual['positions'],policy_phase,plan,world)
                        if safety:
                            raise RuntimeError('stationary recovery pose: '+safety)
                except Exception as safety_error:
                    routine = False
                    reason += '; feedback/recovery check: '+str(safety_error)
            hazardous = motion_started and not routine and not cancelled
            if hazardous:
                rt.stop(reason=reason,hardware_fault=True)
            recovery = rt.recovery.recover(object_id,reason,self.scene,motion_started,self.scene.held is not None)
            rt.report('recovering' if routine and holding else 'failed' if routine else 'blocked',reason)
            return ActionResult(False,reason,object_id,self.scene.state,{'motion_started':motion_started,
                'critical':hazardous or holding or cancelled,'physical_verified':False,'recovery':recovery,
                'recovered':bool(routine and motion_started and not holding),'failure_kind':'OPERATOR_STOP' if cancelled else 'PLANNING_OR_OBSERVATION' if routine else 'SAFETY',
                'hardware_fault':hazardous})
        finally:
            if self.scene.held is None:
                self.scene.release_reservation(object_id)

    def capture_grip_view(self, obj, after_time):
        from .vision import track_color_in_grip
        import cv2
        cameras = self.runtime.cameras
        state = self.robot.read_state();cameras.record_state(state['timestamp'],state['q'],state['gripper'])
        cameras.wait_for_frame('wrist',after_time,self.runtime.check_cancel)
        state = self.robot.read_state();cameras.record_state(state['timestamp'],state['q'],state['gripper'])
        frame = cameras.get_latest_frame('wrist')
        record = self.scene.geometry.projection_for_frame(frame,self.robot)
        camera = np.asarray(record['camera_from_robot']);rotation,_ = cv2.Rodrigues(camera[:3,:3])
        if self.scene.held is None:
            raise ValueError('no confirmed payload attachment')
        tool_object = self.scene.held[1]
        base_object = self.robot.backend.model.frames(state['positions'])['gripper_frame_link']@tool_object
        top = base_object[:3,3]+base_object[:3,2]*obj.dimensions[2]/2
        projected,_ = cv2.projectPoints(top[None],rotation,camera[:3,3],np.asarray(record['camera_matrix']),np.asarray(record['distortion']))
        return track_color_in_grip(frame,obj.color,projected.ravel())

    def verify_pick(self, obj, anchor, after_time):
        self.motion.driver.confirm_retention()
        if anchor is None:
            return False
        observed = self.capture_grip_view(obj,after_time)
        if observed['frame_id']==anchor['frame_id']:
            return False
        visible_attached = np.linalg.norm(np.asarray(observed['center_px'])-anchor['center_px'])<12.
        actual = self.robot.arm.read()[0]
        blocked = self.motion.driver.blocked_grip
        jaw_retained = blocked is not None and abs(actual['gripper']-blocked)<3.
        return bool(jaw_retained and visible_attached)

    def recover_empty_grasp(self, object_id, placement, plan):
        rt = self.runtime
        rt.check_cancel();rt.refresh()
        obj = self.scene.state.objects[object_id]
        if not obj.evidence.get('metric_qualified'):
            raise ValueError('empty grasp recovery needs a newly measured target')
        native = self.scene.native_scene(object_id,placement)
        world = self.motion.make_world(native)
        current = self.robot.read_state()['positions']
        opening = next(p for p in plan.phases if p.name=='enter').joints[-1]['gripper']
        open_path = self.motion.planner.edge(current,dict(current,gripper=opening),world,target=object_id)
        phases = [Phase('open',open_path)]
        cursor = open_path[-1]
        for phase in reversed([p for p in plan.phases if p.name in ('approach','enter')]):
            reverse = tuple(dict(q,gripper=opening) for q in reversed(phase.joints))
            join = self.motion.planner.edge(cursor,reverse[0],world,target=object_id)
            phases.append(Phase('empty_grasp_retreat',tuple(join)+reverse[1:]))
            cursor = reverse[-1]
        recovery = replace(plan,phases=tuple(phases))
        self.motion.check_path(recovery,world)
        for phase in phases:
            self.motion.execute_trajectory(phase,recovery,world)
        rt.report('recovered','Empty jaw confirmed; retraced the checked approach')

    def verify_placement(self, obj, placement):
        before = self.robot.read_state()
        self.runtime.cameras.record_state(before['timestamp'],before['q'],before['gripper'])
        after_motion = time.time()
        for camera in ('overview','wrist'):
            self.runtime.cameras.wait_for_frame(camera,after_motion,self.runtime.check_cancel)
        after = self.robot.read_state()
        self.runtime.cameras.record_state(after['timestamp'],after['q'],after['gripper'])
        self.runtime.refresh()
        matches = [o for o in self.scene.state.objects.values() if o.color==obj.color and o.evidence.get('metric_qualified') and np.linalg.norm(o.pose[:3,3]-placement.object_pose[:3,3]) <= placement.tolerance]
        if len(matches) != 1:
            return False,None
        observed = matches[0]
        if observed.id != obj.id:
            self.scene.state.objects.pop(observed.id)
            observed.id = obj.id
            self.scene.state.objects[obj.id] = observed
        return True,observed.pose
