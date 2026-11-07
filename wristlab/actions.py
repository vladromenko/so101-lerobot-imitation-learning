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

    def rank_first_pick(self,box_id,excluded_colors=()):
        from .grasp import generate_grasps
        from .tasks import placement_in_box
        current = self.robot.read_state()['positions']
        ranking = []
        rejected = {key:reason for (key,reason) in self.runtime.recovery.failures
                    if 'enter is not robust' in reason and 'wrist_camera' in reason}
        for obj in self.scene.state.objects.values():
            self.runtime.check_cancel()
            row = {'object_id':obj.id,'color':obj.color,'admissible':False}
            if obj.id in rejected or obj.color in excluded_colors:
                row['reason'] = 'previously rejected camera/box entry; not retried'
            else:
                placement = placement_in_box(self.scene,obj.id,box_id)
                native = self.scene.native_scene(obj.id,placement)
                world = self.motion.make_world(native)
                candidates = list(generate_grasps(obj,native,self.robot,self.motion.planner,current,world))
                if candidates:
                    point = (np.linalg.inv(native.box.base_from_box)@np.r_[obj.pose[:3,3],1.])[:3]
                    clearance = min(float(np.linalg.norm(np.maximum(np.maximum(lo-point,point-hi),0.)))
                                    for lo,hi,_ in native.box.solid_parts())
                    row.update(admissible=True,grasp_candidates=len(candidates),
                               obstacle_clearance_m=clearance,joint_cost=min(c.score[0] for c in candidates))
                else:
                    row['reason'] = 'no admissible opposing-face grasp IK with rigid accessories'
            ranking.append(row)
        self.runtime.metrics['object_accessibility'] = ranking
        return sorted([r for r in ranking if r['admissible']],key=lambda r:(-r['obstacle_clearance_m'],r['joint_cost']))

    def observe_object(self, object_id):
        self.runtime.refresh()
        obj = self.scene.state.objects[object_id]
        if time.time()-obj.timestamp > .5:
            raise ValueError('fresh target observation is required')
        return obj

    def verify_candidate(self, plan, world, start):
        self.motion.check_entry_transition(plan,world,start)
        return self.motion.preflight_cycle(plan,world)

    def relocate(self, object_id, placement, dry=False, recovery_attempt=0, resume_held=False):
        rt = self.runtime
        started, motion_started = time.monotonic(), False
        executing,last_phase = False,None
        stage = 'observe'
        plan = world = park = None
        aligned,active_refined = False,False
        try:
            obj = self.scene.state.objects[object_id] if resume_held else self.observe_object(object_id)
            if not resume_held:
                self.scene.reserve_object(object_id)
            rt.report(stage,object_id)
            self.robot.read_state()
            reasons = self.motion.qualification_reasons(self.scene.geometry)
            if reasons:
                raise ValueError('; '.join(reasons))
            stage = 'grasp_planning'
            rt.report(stage,object_id)
            if not dry:
                rt.ensure_driver()
            start_state = self.robot.read_state()
            start = start_state['positions']
            choices = [placement]
            if placement.support_surface in self.scene.state.boxes and not resume_held:
                for patch in self.scene.find_free_box_slots(placement.support_surface,obj.dimensions):
                    center = np.r_[np.mean(patch.polygon_xy_m,axis=0),patch.height_m+obj.dimensions[2]/2]
                    if np.linalg.norm(center-placement.object_pose[:3,3])>.015:
                        pose = placement.object_pose.copy();pose[:3,3] = center
                        choices.append(replace(placement,object_pose=pose))
            failures = []
            if resume_held:
                original = rt.resume_original
                native_scene = self.scene.native_scene(object_id,placement)
                native_scene = replace(native_scene,cubes=native_scene.cubes|{object_id:original})
                previous = replace(rt.resume_plan,payload_local=self.motion.driver.payload_local_override)
                plan,native_scene,world = rt.resume_remainder
                start = dict(rt.resume_plan.phases[0].joints[0])
                motion_started = True
            for choice in choices:
                if plan is None:
                    try:
                        obj = self.observe_object(object_id)
                        native_scene = self.scene.native_scene(object_id,choice)
                        world = self.motion.make_world(native_scene)
                        candidate,plan,world,park = choose_grasp(obj,native_scene,self.robot,self.motion.planner,start,rt.check_cancel,world,
                            check_pick=lambda phase,payload:self.motion.check_pick_phase(phase,payload,obj.id,world),
                            stopped_pose=lambda q:self.motion.stop_settled_pose(q),
                            verify_plan=lambda p,w:self.verify_candidate(p,w,start))
                        placement = choice
                    except ValueError as exc:
                        plan = None
                        failures.append(str(exc))
                        rt.metrics.setdefault('placement_refusals',[]).append({
                            'object_id':object_id,'center_m':choice.object_pose[:3,3].tolist(),'reason':str(exc)})
                        rt.report('checking','placement candidate rejected: '+str(exc))
            if plan is None:
                raise ValueError('; '.join(dict.fromkeys(failures)))
            stage = 'path_check'
            rt.last_plan = plan
            if resume_held:
                park = self.motion.preflight_cycle(plan,world,require_pick=False)
            else:
                self.motion.check_entry_transition(plan,world,start)
            if dry:
                phases = (*plan.phases,park)
                return ActionResult(True,'geometric path checked; no actuator commands',object_id,self.scene.state,
                    {'physical_verified':False,'motion_started':False,'scene_revision':plan.scene_revision,
                     'start_feedback_at':start_state['timestamp'],'destination':placement.support_surface,
                     'placement_center_m':plan.placement_center_m.tolist(),
                     'checked_phases':[{'name':p.name,'waypoints':len(p.joints),'carrying':p.carrying,
                                        'start_deg':p.joints[0],'goal_deg':p.joints[-1]} for p in phases]})
            approached = not resume_held and not self.motion.contact_free_passed
            if approached:
                motion_started = True
                executing = True
                self.motion.contact_free_trial(plan,world,start)
                executing = False
            self.motion.plan,self.motion.world = plan,world
            if hasattr(rt,'output'):
                import pickle
                with (rt.output/'active-relocate.pkl').open('wb') as handle:
                    pickle.dump({'plan':plan,'native_scene':native_scene,'object':obj,'placement':placement,'start':start},handle)
            held_verified = False
            wrist_anchor = self.capture_grip_view(obj,time.time()) if resume_held else None
            queue = [p for p in plan.phases if not approached or p.name not in ('open','approach')]
            while queue:
                rt.wait_pause()
                phase = queue.pop(0)
                stage = phase.name
                if stage == 'enter' and not aligned:
                    stage = 'visual_alignment'
                    try:
                        observation = visual_alignment(obj,rt.cameras,self.robot,self.motion,plan,world,self.scene.geometry,
                                                       allow_self_occlusion=object_id in self.scene.reserved)
                    except ValueError:
                        if active_refined or not hasattr(rt,'refine_object'):
                            raise
                        obj = rt.refine_object(object_id,plan,world)
                        active_refined,observation = True,None
                    continuation = observation is not None and observation.get('continuation_only',False)
                    if observation is not None and not continuation:
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
                    if continuation:
                        aligned,stage = True,phase.name
                        rt.report('visual_alignment','reserved overhead plan retained; self-occlusion, fresh wrist identity, no spatial correction')
                    else:
                        rt.refresh()
                        obj = self.scene.state.objects[object_id]
                        native_scene = self.scene.native_scene(object_id,placement)
                        current = self.robot.read_state()['positions']
                        world = self.motion.make_world(native_scene)
                        stage = 'grasp_planning'
                        candidate,plan,world,park = choose_grasp(obj,native_scene,self.robot,self.motion.planner,current,rt.check_cancel,world,
                            check_pick=lambda phase,payload:self.motion.check_pick_phase(phase,payload,obj.id,world),
                            stopped_pose=lambda q:self.motion.stop_settled_pose(q),
                            verify_plan=lambda p,w:self.verify_candidate(p,w,current))
                        self.motion.check_entry_transition(plan,world,current)
                        queue = list(plan.phases)
                        aligned = observation is not None
                        phase = None
                if phase is not None:
                    if stage == 'release' and not held_verified:
                        raise RuntimeError('release refused: independent holding evidence is missing')
                    motion_started = True
                    pending = getattr(rt,'pending_grasp_file',None)
                    if stage in ('lift_off','extract_vertical') and pending is not None and pending.exists():
                        import json
                        checkpoint = json.loads(pending.read_text());checkpoint['lift_performed'] = True
                        pending.write_text(json.dumps(checkpoint,indent=2))
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
                        self.scene.attach_object(object_id,transform,obj)
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
                        self.motion.preflight_cycle(remainder,world,require_pick=False)
                        plan = remainder
                        rt.last_plan = plan
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
                        pending = getattr(rt,'pending_grasp_file',None)
                        if pending is not None and pending.exists():
                            pending.unlink()
                        rt.pending_grasp = None
                        if hasattr(rt,'output'):
                            import pickle
                            placed = next(p.placed_cube for p in queue if p.name=='retract')
                            released = replace(native_scene,cubes=native_scene.cubes|{object_id:placed})
                            checkpoint = {'plan':plan,'native_scene':released,'object':obj,'placement':placement,
                                'start':start,'release_actual':self.robot.read_state()['positions'],
                                'boxes':dict(self.scene.metric_boxes),'calibration_sha256':self.scene.geometry.fingerprint,
                                'boot_id':rt.boot_id}
                            temporary = rt.output/'released-placement.tmp'
                            with temporary.open('wb') as handle:
                                pickle.dump(checkpoint,handle)
                            temporary.replace(rt.output/'released-placement.pkl')
            stage = 'verify_placement'
            verify_started = time.monotonic()
            observed_after = time.time()
            verified,observed = self.verify_placement(obj,placement)
            rt.metrics['verification_s'] = (rt.metrics.get('verification_s',0.) or 0.)+time.monotonic()-verify_started
            if not verified and recovery_attempt < rt.recovery.max_attempts:
                with self.scene.lock:
                    candidates = [o for o in self.scene.state.objects.values() if o.color==obj.color and self.fresh_metric_observation(o,observed_after)]
                if len(candidates)==1:
                    self.scene.release_reservation(object_id)
                    retry = self.relocate(candidates[0].id,placement,recovery_attempt=recovery_attempt+1)
                    if not retry.evidence.get('motion_started'):
                        retry.evidence.update(recovered=True)
                    retry.evidence.update(motion_started=True,previous_retreat_completed=True)
                    return retry
            if not verified:
                raise ValueError('object was not observed at placement within tolerance')
            self.scene.detach_object(object_id,observed)
            if hasattr(rt,'record_action_measurement'):
                measured = self.scene.state.objects[object_id]
                rt.record_action_measurement('placement',object_id,{'requested_pose_m':placement.object_pose.tolist(),
                    'observed_pose_m':observed.tolist(),'error_m':(observed[:3,3]-placement.object_pose[:3,3]).tolist(),
                    'position_sigma_m':measured.evidence['position_sigma_m']})
            parked = False
            placed_cube = self.scene.native_object(object_id)
            observed_scene = replace(native_scene,cubes=native_scene.cubes|{object_id:placed_cube})
            observed_world = self.motion.make_world(observed_scene)
            # Park collision policies may override the world's target cube
            # with the release phase's geometry. Rebind both to the observation.
            observed_plan = replace(plan,phases=tuple(replace(p,placed_cube=placed_cube) if p.placed_cube is not None else p
                                                       for p in plan.phases))
            self.motion.plan,self.motion.world = observed_plan,observed_world
            if rt.mode=='relocate' and rt.park_qualified_file.exists():
                stage = 'park'
                rt.park_arm(observed_plan,observed_world)
                parked = True
            rt.physical_relocate_count += 1
            rt.metrics['cycle_s'] = time.monotonic()-started
            return ActionResult(True,'relocated and observed at destination',object_id,self.scene.state,
                {'physical_verified':True,'motion_started':True,'placement_tolerance_m':placement.tolerance,
                 'retreat_completed':True,'parked':parked,'park_pending_attended_trial':rt.mode=='relocate' and not parked})
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

    def capture_grip_view(self, obj, after_time, camera_id=None):
        from .vision import track_color_in_grip
        import cv2
        cameras = self.runtime.cameras
        state = self.robot.read_state();cameras.record_state(state['timestamp'],state['q'],state['gripper'])
        cameras.wait_for_frame('wrist',after_time,self.runtime.check_cancel)
        state = self.robot.read_state();cameras.record_state(state['timestamp'],state['q'],state['gripper'])
        if self.motion.grasp_mode=='overhead_grasp':
            def wrist_image():
                frame = cameras.get_latest_frame('wrist')
                center = np.array(frame.image.shape[1::-1])/2
                result = track_color_in_grip(frame,obj.color,center,radius_px=float(np.linalg.norm(frame.image.shape[:2])))
                return dict(result,relative_error_px=result['center_px'],camera_id='wrist',spatial_update=False)
            if camera_id=='wrist':
                return wrist_image()
            if self.scene.held is None:
                raise ValueError('no force-confirmed payload attachment')
            base_object = self.robot.backend.model.frames(state['positions'])['gripper_frame_link']@self.scene.held[1]
            top = base_object[:3,3]+base_object[:3,2]*obj.dimensions[2]/2
            predicted = self.scene.geometry.project(top[None])[0]
            cameras.wait_for_frame('overview',after_time,self.runtime.check_cancel)
            frame = cameras.get_latest_frame('overview')
            try:
                result = track_color_in_grip(frame,obj.color,predicted,radius_px=45.)
                return dict(result,relative_error_px=(np.asarray(result['center_px'])-predicted).tolist(),camera_id='overview')
            except ValueError:
                return wrist_image()
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
        observed = self.capture_grip_view(obj,after_time,camera_id=anchor.get('camera_id'))
        if observed['frame_id']==anchor['frame_id']:
            return False
        field = 'relative_error_px' if self.motion.grasp_mode=='overhead_grasp' else 'center_px'
        visible_attached = np.linalg.norm(np.asarray(observed[field])-anchor[field])<12.
        if observed.get('camera_id')!=anchor.get('camera_id'):
            return False
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

    def fresh_metric_observation(self, candidate, after_time):
        evidence = candidate.evidence
        overview_id = self.scene.state.evidence.get('overview_frame_id')
        overview = overview_id is not None and evidence.get('visible_overview') is True and evidence.get('frame_id')==overview_id
        wrist = evidence.get('wrist_observation',{})
        wrist_id = self.scene.state.evidence.get('wrist_frame_id')
        wrist_metric = wrist_id is not None and self.scene.geometry.wrist is not None and wrist.get('spatial_update') is True and wrist.get('frame_id')==wrist_id
        return (evidence.get('metric_qualified') and candidate.timestamp>=after_time and
                evidence.get('spatial_timestamp',candidate.timestamp)>=after_time and
                candidate.status not in ('held','occluded') and (overview or wrist_metric))

    def verify_placement(self, obj, placement):
        before = self.robot.read_state()
        self.runtime.cameras.record_state(before['timestamp'],before['q'],before['gripper'])
        after_motion = time.time()
        for camera in ('overview','wrist'):
            self.runtime.cameras.wait_for_frame(camera,after_motion,self.runtime.check_cancel)
        after = self.robot.read_state()
        self.runtime.cameras.record_state(after['timestamp'],after['q'],after['gripper'])
        self.runtime.refresh()
        with self.scene.lock:
            matches = [o for o in self.scene.state.objects.values() if o.color==obj.color and self.fresh_metric_observation(o,after_motion) and
                       np.linalg.norm(o.pose[:3,3]-placement.object_pose[:3,3])<=placement.tolerance]
            if len(matches)!=1:
                return False,None
            observed = matches[0]
            if observed.id!=obj.id:
                old_id = observed.id
                self.scene.state.objects.pop(old_id)
                self.scene.track_memory.pop(old_id,None)
                metric = self.scene.metric_objects.pop(old_id,None)
                if metric is not None:
                    self.scene.metric_objects[obj.id] = metric
                observed.id = obj.id
                self.scene.state.objects[obj.id] = observed
                for box in self.scene.state.boxes.values():
                    box.occupied_slots = [obj.id if key==old_id else key for key in box.occupied_slots]
            self.scene.track_memory[obj.id] = observed
            return True,observed.pose.copy()
