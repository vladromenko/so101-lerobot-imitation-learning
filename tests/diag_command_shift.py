"""Plan on the live scene (from a saved scan), then express the gravity-compensation command shift in the gripper frame at every phase's waypoints."""
import json, sys, time
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
import extract_from_box as cli
scan = json.load(open(sys.argv[1])); frame = cv2.imread(scan['artifacts'] + '/frame.jpg')
if not any(d['label'] == 'box_opening' for d in scan['detections']):
    scan['detections'].append(estimate_opening(frame, scan)[0])
config = load_config(ROOT / 'config/default.json'); cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config.robot['calibration'], table_z_m=0., offsets=config.robot['kinematic_offsets_deg'])
params = SceneParameters(wall_clearance_m=.003)
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='o', ids=IdentityTracker())
ready = cli.choose_ready(backend, replace(scene, captured_at=time.time()))
plan, trial = cli.plan_scene(TunedExtractionPlanner(backend), scene, patches, ready, print, params)
sag = 3.7
for phase in plan.phases:
    if phase.name not in ('enter', 'grasp'):
        continue
    for tag, q in (('first', phase.joints[0]), ('last', phase.joints[-1])):
        cmd = backend.driver_command_goal(q, sag)
        f0 = backend.model.frames(q)['gripper_frame_link']; f1 = backend.model.frames(cmd)['gripper_frame_link']
        d_local = (f1[:3, 3] - f0[:3, 3]) @ f0[:3, :3]
        rot = np.degrees(np.arccos(np.clip((np.trace(f0[:3, :3].T @ f1[:3, :3]) - 1) / 2, -1, 1)))
        print(phase.name, tag, 'shift in gripper frame (mm) x,y,z', np.round(d_local * 1000, 1), 'rot deg', round(rot, 2), 'gripper', round(q['gripper'], 1))
