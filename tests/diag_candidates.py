"""Why does each grasp candidate fail? (offline, no motors)"""
import json, sys, time, collections
from dataclasses import replace
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'
from so101_pick_place.kinematics import So101ModelPlanner
from box_extract.planning import TunedExtractionPlanner as ExtractionPlanner
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld
from box_extract.scene_builder import *

scan = json.loads(Path(sys.argv[1]).read_text()); frame = cv2.imread(sys.argv[2])
sig, clr, flo = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
config = json.loads((ROOT / 'config/default.json').read_text())
cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config['robot']['calibration'], table_z_m=cal['table_z_m'], offsets=config['robot']['kinematic_offsets_deg'])
params = SceneParameters(position_sigma_m=sig, vertical_sigma_m=sig, wall_clearance_m=clr, floor_interval_m=(0., flo))
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='probe', ids=IdentityTracker())
scene = replace(scene, placement=patches[0], captured_at=time.time())
import os
if os.environ.get('SHIFT'):            # what-if: the whole box (and its cubes) stands elsewhere
    dx, dy = map(float, os.environ['SHIFT'].split(','))
    t = np.array([dx, dy, 0.])
    transform = scene.box.base_from_box.copy(); transform[:3, 3] += t
    scene = replace(scene, box=replace(scene.box, base_from_box=transform),
                    cubes={k: replace(c, center_m=c.center_m + t, vertices_m=c.vertices_m + t) if inside_box(c, scene.box) else c for k, c in scene.cubes.items()})
READY = None
for z in (.17, .15, .13, .11, .09):
    for pitch in (-90., -75., -60.):
        try:
            READY = backend.model.ik(.17, -.10, z, pitch=pitch, roll=0., gripper=40.).deg; break
        except Exception: pass
    if READY: print('ready z', z, 'pitch', pitch); break
print('ready pose', {k: round(v, 1) for k, v in READY.items()}, 'problems', backend.model.problems(backend.model.pose(READY)))
current = READY
planner = ExtractionPlanner(backend)
world = ExtractionCollisionWorld(backend, scene)
inv = np.linalg.inv(scene.box.base_from_box)
for tid, cube in scene.cubes.items():
    loc = inv[:3, :3] @ cube.center_m + inv[:3, 3]
    if not inside_box(cube, scene.box): continue
    print('TARGET', tid, 'local xyz mm', np.round(loc * 1000, 1), 'inner', np.round(scene.box.inner_min_xy * 1000), np.round(scene.box.inner_max_xy * 1000))
    stats = collections.Counter()
    n = 0
    for grasp, closed in planner.grasp_candidates(scene, cube, current):
        n += 1
        frame_g = backend.model.frames(grasp)['gripper_frame_link']
        tilt = np.rad2deg(np.arccos(np.clip(-frame_g[2, 2], -1, 1)))
        policy = {'target': tid, 'jaw_contact': True}
        at_pose = world.reason(grasp, **policy)
        try:
            planner._candidate(scene, current, tid, cube, grasp, closed, world)
            res = 'OK'
        except Exception as exc:
            res = str(exc)[:110]
        stats[res] += 1
        if n <= 6:
            print(f'  cand {n}: tilt {tilt:.0f} deg, closing_x {np.round(frame_g[:3,0],2)}, at-pose: {at_pose}, result: {res}')
    print('  total candidates', n, dict(stats))
