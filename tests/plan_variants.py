"""Which constraint blocks the plan for the current scene? One planner variant per run (env: SAG=0/1, TRACK=deg, GAP=mm, MARGIN=mm)."""
import json, os, sys, time
from dataclasses import replace
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'scripts'))
ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'; sys.path.insert(0, str(ROOT / 'scripts'))
from so101_pick_place.config import load_config
from so101_pick_place.kinematics import So101ModelPlanner
from box_extract.scene_builder import *
from box_extract.planning import TunedExtractionPlanner
from box_extract.opening_fallback import estimate_opening
from box_extract.driver import read_command_sag
import extract_from_box as cli
scan = json.load(open(sys.argv[1])); frame = cv2.imread(scan['artifacts'] + '/frame.jpg')
if not any(d['label'] == 'box_opening' for d in scan['detections']):
    scan['detections'].append(estimate_opening(frame, scan)[0])
config = load_config(ROOT / 'config/default.json'); cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config.robot['calibration'], table_z_m=0., offsets=config.robot['kinematic_offsets_deg'])
params = SceneParameters(wall_clearance_m=.003, flap_planning_margin_m=float(os.environ.get('MARGIN', 24)) / 1000., planning_wall_clearance_m=.004)
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='o', ids=IdentityTracker())
ready = cli.choose_ready(backend, replace(scene, captured_at=time.time()))
planner = TunedExtractionPlanner(backend)
planner.command_sag = read_command_sag(config, scan['robot_joint_deg'], HERE / 'tmp/pv') if os.environ.get('SAG', '1') == '1' else None
planner.tracking_error_deg = float(os.environ.get('TRACK', 1.5)); planner.fixed_jaw_gap_m = float(os.environ.get('GAP', 2.5)) / 1000.
print('variant', {k: os.environ.get(k) for k in ('SAG', 'TRACK', 'GAP', 'MARGIN')}, 'cubes inside', info['inside_box'], flush=True)
t = time.time()
try:
    plan, trial = cli.plan_scene(planner, scene, patches, ready, print, params, max_patches=1)
    print('FEASIBLE in', round(time.time() - t), 's', [(p.name, len(p.joints)) for p in plan.phases])
except Exception as exc:
    print('INFEASIBLE in', round(time.time() - t), 's:', str(exc)[:600])
