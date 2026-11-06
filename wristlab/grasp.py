"""One grasp candidate source for every task: existing measured jaw geometry."""
import json
from pathlib import Path
import numpy as np
from so101_pick_place.vision.wrist_servo import WristVisualServo, ServoCalibration, ServoLimits, WristObservation
from so101_pick_place.experiments.box_extraction.planner import Phase
from .types import GraspCandidate


def generate_grasps(object_state, native_scene, robot, planner, current, world):
    cube = native_scene.cubes[object_state.id]
    for grasp,closed in planner.grasp_candidates(native_scene,cube,current):
        reason = world.reason(grasp,target=object_state.id,jaw_contact=True)
        if reason is None:
            pose = robot.backend.model.frames(grasp)['gripper_frame_link']
            pregrasp = pose.copy()
            pregrasp[2,3] += max(.025,native_scene.box.obstacle_top_m-cube.center_m[2]+cube.side_m/2+.02)
            q = robot.from_native(grasp)
            cost = float(np.linalg.norm(q-robot.from_native(current)))
            # Native closure is retained, instead of inventing a percent-to-metre scale.
            yield GraspCandidate(pose,float(cube.side_m+.008),pregrasp,np.array([0.,0.,1.]),q,
                float(planner.fixed_jaw_gap_m),(cost,),native=(grasp,closed))


def rank_grasps(candidates):
    return sorted(candidates,key=lambda c:c.score)


def choose_grasp(obj, scene, robot, planner, current, check_cancel, world):
    failures = []
    for candidate in generate_grasps(obj,scene,robot,planner,current,world):
        check_cancel()
        grasp,closed = candidate.native
        try:
            plan = planner._candidate(scene,current,obj.id,scene.cubes[obj.id],grasp,closed,world)
            return candidate,plan,world
        except ValueError as exc:
            failures.append(str(exc))
    raise ValueError('no full-arm grasp/approach/lift/placement path: '+('; '.join(failures[-3:]) or 'no admissible opposing-face grasp'))


def visual_alignment(object_state, cameras, robot, motion, plan, world, calibration_path):
    record = json.loads(Path(calibration_path).read_text())
    reference_q = np.asarray(record['reference_q_rad'])
    current = robot.read_state()
    if np.max(np.abs(current['q']-reference_q)) > float(record['validated_joint_radius_rad']):
        raise ValueError('wrist image Jacobian does not cover this measured pose')
    if record.get('current_mount_independently_validated') is not True:
        raise ValueError('wrist servo was not qualified for this attachment')
    servo = WristVisualServo(ServoCalibration.from_record(record),ServoLimits())
    anchor = np.asarray(record['target_px'])
    from .vision import detect_cubes
    last_frame_id = None
    for iteration in range(servo.limits.max_iterations+1):
        motion.check_cancel()
        frame = cameras.get_latest_frame('wrist')
        if frame.frame_id == last_frame_id:
            raise ValueError('wrist servo received a repeated frame')
        last_frame_id = frame.frame_id
        if frame.joints is None or frame.capture_interval[1]-frame.capture_interval[0] > .1:
            raise ValueError('wrist frame is not paired with measured joints')
        detections = [d for d in detect_cubes(frame) if d['color']==object_state.color]
        detection = min(detections,key=lambda d:np.linalg.norm(np.asarray(d['center_px'])-anchor)) if detections else None
        observation = WristObservation(int(frame.frame_id.rsplit('-',1)[-1]),time_age(frame),None if detection is None else tuple(detection['center_px']),0. if detection is None else detection['confidence'])
        decision = servo.decide(observation,iteration)
        if decision.state == 'ALIGNED':
            return {'center_px':list(detection['center_px']),'frame_id':frame.frame_id}
        if decision.state != 'CORRECT':
            raise ValueError(decision.reason)
        q = robot.arm.read()[0]
        goal = motion.planner.translated(q,[*decision.correction_xy_m,0.])
        path = motion.planner.edge(q,goal,world,target=object_state.id,jaw_contact=True)
        phase = Phase('visual_align',path,jaw_contact=True)
        motion.execute_trajectory(phase,plan,world)
    raise ValueError('bounded wrist alignment exhausted')


def time_age(frame):
    import time
    return time.time()-frame.capture_interval[0]
