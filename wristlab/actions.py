"""The universal relocate operation; all experiments share this physical path."""
from dataclasses import replace
import json
import time
import numpy as np
from so101_pick_place.experiments.box_extraction.planner import Phase
from .grasp import choose_grasp, visual_alignment
from .types import ActionResult
from .vision import detect_cubes


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

    def relocate(self, object_id, placement, dry=False):
        rt = self.runtime
        started, motion_started = time.monotonic(), False
        stage = 'observe'
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
            rt.metrics['grasp_planning_s'] = time.monotonic()-started
            self.motion.check_path(plan,world)
            rt.last_plan = plan
            if dry:
                return ActionResult(True,'geometric path checked; no actuator commands',object_id,self.scene.state,
                                    {'physical_verified':False,'motion_started':False})
            if not self.motion.contact_free_passed:
                motion_started = True
                self.motion.contact_free_trial(plan,world,start)
            self.motion.plan,self.motion.world = plan,world
            held_verified = False
            wrist_anchor = None
            queue = list(plan.phases)
            while queue:
                rt.wait_pause()
                phase = queue.pop(0)
                stage = phase.name
                if stage == 'grasp':
                    wrist_anchor = visual_alignment(obj,rt.cameras,self.robot,self.motion,plan,world,rt.root/'runtime/local-servo.json')
                    actual = self.robot.arm.read()[0]
                    closure = tuple(dict(actual,gripper=q['gripper']) for q in phase.joints)
                    phase = replace(phase,joints=closure)
                if stage == 'release' and not held_verified:
                    raise RuntimeError('release refused: independent holding evidence is missing')
                motion_started = True
                self.motion.execute_trajectory(phase,plan,world)
                if stage == 'grasp':
                    local = self.motion.driver.confirm_grasp()
                    actual = self.robot.arm.read()[0]
                    transform = np.linalg.inv(self.robot.backend.model.frames(actual)['gripper_frame_link']) @ obj.pose
                    self.scene.attach_object(object_id,transform)
                    # Post-contact attachment is derived from measured encoders.
                    adjusted = replace(plan,payload_local=local)
                    remainder,new_scene = self.motion.planner.replan_held(native_scene,actual,adjusted,native_scene.cubes[object_id],enforce_fresh=False)
                    world = self.motion.make_world(new_scene)
                    self.motion.check_path(remainder,world)
                    plan = remainder
                    queue = list(remainder.phases)
                elif stage == 'extract_vertical':
                    held_verified = self.verify_pick(obj,wrist_anchor)
                    if not held_verified:
                        raise RuntimeError('holding not confirmed by jaw feedback and wrist observation')
                elif stage == 'release':
                    self.motion.driver.release_confirmed()
                    self.scene.held = None
            stage = 'verify_placement'
            verified,observed = self.verify_placement(obj,placement)
            if not verified:
                raise RuntimeError('object was not observed at placement within tolerance')
            self.scene.detach_object(object_id,observed)
            self.motion.return_to_start(start,plan,world)
            rt.metrics['cycle_s'] = time.monotonic()-started
            return ActionResult(True,'relocated and observed at destination; returned to working pose',object_id,self.scene.state,
                {'physical_verified':True,'motion_started':True,'placement_tolerance_m':placement.tolerance})
        except Exception as exc:
            reason = f'{stage}: {type(exc).__name__}: {exc}'
            if motion_started:
                self.robot.stop_motion()
            recovery = rt.recovery.recover(object_id,reason,self.scene,motion_started,self.scene.held is not None)
            rt.report('blocked',reason)
            return ActionResult(False,reason,object_id,self.scene.state,{'motion_started':motion_started,
                'critical':motion_started or stage=='observe','physical_verified':False,'recovery':recovery})
        finally:
            if self.scene.held is None:
                self.scene.release_reservation(object_id)

    def verify_pick(self, obj, anchor):
        self.motion.driver.confirm_retention()
        if anchor is None:
            return False
        frame = self.runtime.cameras.get_latest_frame('wrist')
        matches = [d for d in detect_cubes(frame) if d['color']==obj.color]
        visible_attached = any(np.linalg.norm(np.asarray(d['center_px'])-anchor['center_px']) < 12. for d in matches)
        actual = self.robot.arm.read()[0]
        blocked = self.motion.driver.blocked_grip
        jaw_retained = blocked is not None and abs(actual['gripper']-blocked) < 3.
        return bool(jaw_retained and visible_attached)

    def verify_placement(self, obj, placement):
        self.runtime.refresh()
        matches = [o for o in self.scene.state.objects.values() if o.color==obj.color and np.linalg.norm(o.pose[:3,3]-placement.object_pose[:3,3]) <= placement.tolerance]
        if len(matches) != 1:
            return False,None
        observed = matches[0]
        if observed.id != obj.id:
            self.scene.state.objects.pop(observed.id)
            observed.id = obj.id
            self.scene.state.objects[obj.id] = observed
        return True,observed.pose
