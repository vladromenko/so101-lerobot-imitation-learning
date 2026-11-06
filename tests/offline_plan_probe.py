"""Plan (no motors, no server) on a saved scan: which patches/sigmas admit an extraction plan?"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'
from so101_pick_place.kinematics import So101ModelPlanner
from so101_pick_place.experiments.box_extraction.planner import ExtractionPlanner
from box_extract.scene_builder import *

scan = json.loads(Path(sys.argv[1]).read_text()); frame = cv2.imread(sys.argv[2])
position_sigma = float(sys.argv[3]) if len(sys.argv) > 3 else .002
clearance = float(sys.argv[4]) if len(sys.argv) > 4 else .006
floor_hi = float(sys.argv[6]) if len(sys.argv) > 6 else .003
config = json.loads((ROOT / 'config/default.json').read_text())
cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config['robot']['calibration'], table_z_m=cal['table_z_m'], offsets=config['robot']['kinematic_offsets_deg'])
params = SceneParameters(position_sigma_m=position_sigma, vertical_sigma_m=position_sigma, wall_clearance_m=clearance, floor_interval_m=(0., floor_hi))
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='probe', ids=IdentityTracker())
current = scan['robot_joint_deg']
planner = ExtractionPlanner(backend)
from so101_pick_place.experiments.box_extraction.scene import localization_margin
for k, c in scene.cubes.items():
    print(k, 'localization margin mm', round(1000 * localization_margin(c), 1))
for patch in patches[:int(sys.argv[5]) if len(sys.argv) > 5 else 3]:
    trial = replace(scene, placement=patch, captured_at=time.time())
    t = time.time()
    try:
        plan = planner.plan(trial, current)
        print('PLAN OK target', plan.target_id, 'patch', patch.evidence['center_xy_m'], 'phases', [(p.name, len(p.joints)) for p in plan.phases], 'secs', round(time.time() - t, 1))
        break
    except Exception as exc:
        print('plan failed for patch', patch.evidence['center_xy_m'], type(exc).__name__, str(exc)[:600], 'secs', round(time.time() - t, 1))
