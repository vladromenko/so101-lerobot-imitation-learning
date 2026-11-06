"""Box-nudge handling offline: rim matching with the arm over the box, and the corrected pending path in a shifted world."""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'scripts'))
ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'; sys.path.insert(0, str(ROOT / 'scripts'))
from so101_pick_place.config import load_config
from so101_pick_place.kinematics import So101ModelPlanner
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld
from box_extract.scene_builder import *
from box_extract.planning import TunedExtractionPlanner
from box_extract.opening_fallback import estimate_opening
from box_extract.perception import OverviewPerception
from box_extract.driver import read_command_sag
from box_extract.shift import shift_scene, correct_pending
import extract_from_box as cli

config = load_config(ROOT / 'config/default.json'); cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config.robot['calibration'], table_z_m=0., offsets=config.robot['kinematic_offsets_deg'])
params = SceneParameters(wall_clearance_m=.003)
scan = json.load(open(sys.argv[1])); frame = cv2.imread(scan['artifacts'] + '/frame.jpg')
opening, _ = estimate_opening(frame, scan); scan['detections'].append(opening)
perception = OverviewPerception('x', cal, backend, params)
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='o', ids=IdentityTracker())
perception.reference = perception._reference(scan, scene, frame)
# 1. rim matching: shifted frame, then the arm over the box
none = np.zeros(frame.shape[:2], np.uint8)
for dx, dy in ((0, 0), (6, 2), (-10, 5), (20, -8)):
    M = np.float32([[1, 0, dx], [0, 1, dy]]); r = perception.rim_shift(cv2.warpAffine(frame, M, frame.shape[1::-1]), none)
    print('synthetic shift', (dx, dy), '->', r['shift_px'], round(r['score'], 2), '| robot-frame mm', np.round(perception.px_to_xy(r['shift_px']) * 1000, 1))
plan_json = json.load(open(sys.argv[2])); joints = plan_json['phases'][1]['joints'][-1]
arm_frame = cv2.imread(sys.argv[3]); occluder = cv2.dilate(robot_mask(backend, joints, cal, arm_frame.shape), np.ones((61, 61), np.uint8))
r = perception.rim_shift(arm_frame, occluder); print('arm over the box (real frame):', r)
# 2. corrected pending path in a shifted world
ready = cli.choose_ready(backend, replace(scene, captured_at=time.time()))
planner = TunedExtractionPlanner(backend); planner.command_sag = read_command_sag(config, scan['robot_joint_deg'], HERE / 'tmp/shift_probe')
plan, trial = cli.plan_scene(planner, scene, patches, ready, print, params)
pending = [p for p in plan.phases if p.name in ('approach', 'enter', 'grasp')]
for d in ((.006, -.004), (-.010, .008), (.015, .012), (.024, 0.)):
    world = ExtractionCollisionWorld(backend, shift_scene(trial, d)); t = time.time()
    try:
        out = correct_pending(planner, world, pending, d, plan.target_id)
        print('shift mm', np.round(np.array(d) * 1000, 1), 'OK', [(p.name, len(p.joints)) for p in out], round(time.time() - t, 1), 's')
    except ValueError as exc:
        print('shift mm', np.round(np.array(d) * 1000, 1), 'REFUSED:', exc)
