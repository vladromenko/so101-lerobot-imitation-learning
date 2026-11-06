import json, sys, time
from dataclasses import replace
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'
from so101_pick_place.kinematics import So101ModelPlanner
from so101_pick_place.experiments.box_extraction.planner import ExtractionPlanner
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld
from box_extract.scene_builder import *
scan = json.loads(Path(sys.argv[1]).read_text()); frame = cv2.imread(sys.argv[2])
config = json.loads((ROOT / 'config/default.json').read_text()); cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config['robot']['calibration'], table_z_m=0., offsets=config['robot']['kinematic_offsets_deg'])
params = SceneParameters(position_sigma_m=.002, vertical_sigma_m=.002, wall_clearance_m=.003, floor_interval_m=(0., .003))
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='p', ids=IdentityTracker())
current = scan['robot_joint_deg']; planner = ExtractionPlanner(backend); model = backend.model
cube = next(c for c in scene.cubes.values() if inside_box(c, scene.box))
grasp, closed = next(iter(planner.grasp_candidates(scene, cube, current)))
print('grasp joints', {k: round(v, 1) for k, v in grasp.items()}, 'ranges', np.round(planner.limits, 1).tolist())
frame_g = model.frames(grasp)['gripper_frame_link']; local = (cube.vertices_m - frame_g[:3, 3]) @ frame_g[:3, :3]
center = frame_g[:3, 3] + frame_g[:3, :3] @ local.mean(0)
print('payload centre', np.round(center, 3), 'radius', round(float(np.hypot(*center[:2])), 3))
prev = grasp
for dz in (.02, .04, .06, .08, .10, .12):
    target = center + [0, 0, dz]
    seed = planner.vector(prev)
    def residual(q):
        a = model.frames(planner.joints(q, grasp['gripper']))['gripper_frame_link']
        return np.r_[(a[:3, 3] + a[:3, :3] @ local.mean(0) - target) * 1000., .03 * (q - seed)]
    from scipy.optimize import least_squares
    fit = least_squares(residual, seed, bounds=(planner.limits[:, 0], planner.limits[:, 1]), max_nfev=100)
    j = planner.joints(fit.x, grasp['gripper'])
    pose = model.pose(j)
    a = model.frames(j)['gripper_frame_link']
    print(f'dz {dz:.2f}: pos err mm {np.linalg.norm(residual(fit.x)[:3]):.2f} joints {np.round(fit.x,1).tolist()} tilt {np.rad2deg(np.arccos(np.clip(-a[2,2],-1,1))):.0f} problems {model.problems(pose)}')
    prev = j

print('--- full planner._candidate for each grasp candidate with full exception text')
world = ExtractionCollisionWorld(backend, replace(scene, placement=patches[0], captured_at=time.time()))
sc = replace(scene, placement=patches[0], captured_at=time.time())
for grasp, closed in planner.grasp_candidates(sc, cube, current):
    try:
        planner._candidate(sc, current, 'cube-1', cube, grasp, closed, world); print('OK')
    except Exception as e:
        print(str(e)[:700])
