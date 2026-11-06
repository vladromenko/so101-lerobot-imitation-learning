"""One grasp candidate source for every task: existing measured jaw geometry."""
import numpy as np
import time
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
    from box_extract.planning import flush_local
    failures = []
    metrics = getattr(planner,'timings',{})
    started = time.perf_counter()
    try:
        for candidate in generate_grasps(obj,scene,robot,planner,current,world):
            check_cancel()
            grasp,closed = candidate.native
            begin = time.perf_counter()
            try:
                # Reject an unreachable destination before constructing and
                # checking hundreds of approach/closure/lift waypoints.
                cube = scene.cubes[obj.id]
                frame = robot.backend.model.frames(grasp)['gripper_frame_link']
                local = flush_local((cube.vertices_m-frame[:3,3])@frame[:3,:3])
                held = dict(grasp,gripper=min(closed,planner.last_clamp_deg))
                patch = scene.placement
                center = np.r_[np.mean(patch.polygon_xy_m,axis=0),
                               patch.height_m+patch.uncertainty_m+cube.side_m/2+.001]
                down = planner.placement_pose(held,local,cube,center)
                for pose in (down,planner._command_pose(down)):
                    reason = world.reason(pose,target=obj.id,jaw_contact=True,payload_local=local,support_contact=True)
                    if reason:
                        raise ValueError('placement pose: '+reason)
                plan = planner._candidate(scene,current,obj.id,scene.cubes[obj.id],grasp,closed,world)
                return candidate,plan,world
            except ValueError as exc:
                failures.append(str(exc))
                metrics['candidate_failures'] = failures[-12:]
            finally:
                metrics['motion_planning_s'] = (metrics.get('motion_planning_s',0.) or 0.)+time.perf_counter()-begin
        raise ValueError('no full-arm grasp/approach/lift/placement path: '+('; '.join(failures[-3:]) or 'no admissible opposing-face grasp'))
    finally:
        metrics['grasp_planning_s'] = (metrics.get('grasp_planning_s',0.) or 0.)+time.perf_counter()-started


def visual_alignment(object_state, cameras, robot, motion, plan, world, geometry):
    """Align at pregrasp, before fingers descend. No full-scene micro replans."""
    import time
    import cv2
    from .vision import detect_cubes
    from so101_pick_place.vision.wrist_camera_geometry import camera_projection
    if motion.grasp_mode=='overhead_grasp':
        state = robot.read_state()
        cameras.record_state(state['timestamp'],state['q'],state['gripper'])
        cameras.wait_for_frame('overview',time.time(),motion.check_cancel)
        frame = cameras.get_latest_frame('overview')
        matches = [d for d in detect_cubes(frame) if d['color']==object_state.color]
        if len(matches)!=1:
            raise ValueError('overhead target is missing or ambiguous')
        measured = geometry.estimate_cube_pose(matches[0],frame,object_state.evidence['support_z_m'])
        if not measured.evidence.get('metric_qualified'):
            raise ValueError('overhead target no longer has a complete supported top face')
        return {'center_px':matches[0]['center_px'],'frame_id':frame.frame_id,
                'observed_top_m':(measured.pose[:3,3]+[0.,0.,measured.dimensions[2]/2]).tolist(),
                'grasp_mode':'overhead_grasp','wrist_refinement':False}
    if geometry.wrist is None:
        raise ValueError('qualified wrist hand-eye calibration is required')
    reference = next(p for p in plan.phases if p.name=='enter').joints[0]
    top = object_state.pose[:3,3]+[0.,0.,object_state.dimensions[2]/2]
    def project(q):
        record = camera_projection(geometry.wrist,robot.backend.model,q)
        camera = np.asarray(record['camera_from_robot'])
        rotation,_ = cv2.Rodrigues(camera[:3,:3])
        uv,_ = cv2.projectPoints(top[None],rotation,camera[:3,3],np.asarray(record['camera_matrix']),np.asarray(record['distortion']))
        return uv.ravel()
    desired = project(reference)
    last_frame_id = None
    after_motion = time.time()
    for iteration in range(13):
        motion.check_cancel()
        before = robot.read_state();cameras.record_state(before['timestamp'],before['q'],before['gripper'])
        cameras.wait_for_frame('wrist',after_motion,motion.check_cancel)
        after = robot.read_state();cameras.record_state(after['timestamp'],after['q'],after['gripper'])
        frame = cameras.get_latest_frame('wrist')
        if frame.frame_id==last_frame_id or frame.joints is None or frame.motion_during_exposure_rad is None or frame.motion_during_exposure_rad>.002:
            raise ValueError('wrist frame is repeated or not paired with stationary feedback')
        last_frame_id = frame.frame_id
        detections = [d for d in detect_cubes(frame) if d['color']==object_state.color]
        matches = sorted(detections,key=lambda d:np.linalg.norm(np.asarray(d['center_px'])-desired))
        if not matches or (len(matches)>1 and np.linalg.norm(np.asarray(matches[1]['center_px'])-desired)<20.):
            raise ValueError('wrist target is missing or ambiguous')
        measured = matches[0]
        current = after['positions']
        base = project(current)
        jacobian = np.column_stack([(project(motion.planner.translated(current,delta))-base)/.001 for delta in ([.001,0.,0.],[0.,.001,0.])])
        singular = np.linalg.svd(jacobian,compute_uv=False)
        if singular[-1]<100. or singular[0]/singular[-1]>25.:
            raise ValueError('wrist image Jacobian is ill-conditioned')
        limits = ServoLimits(max_frame_age_s=max(.15,cameras.exposure_s.get('wrist',0.)+.2))
        servo = WristVisualServo(ServoCalibration(tuple(desired),jacobian),limits)
        observation = WristObservation(iteration,time.time()-frame.capture_interval[0],tuple(measured['center_px']),measured['confidence'])
        decision = servo.decide(observation,iteration)
        if decision.state=='ALIGNED':
            ray = geometry.pixel_ray_in_base(measured['center_px'],frame,robot)
            observed_top = geometry.intersect_ray_with_plane(ray,([0.,0.,1.],-top[2]))
            return {'center_px':measured['center_px'],'frame_id':frame.frame_id,'observed_top_m':observed_top.tolist()}
        if decision.state!='CORRECT':
            raise ValueError(decision.reason)
        goal = motion.planner.translated(current,[*decision.correction_xy_m,0.])
        path = motion.planner.edge(current,goal,world,target=object_state.id)
        motion.execute_trajectory(Phase('visual_align',path),plan,world)
        after_motion = time.time()
    raise ValueError('bounded wrist alignment exhausted')
