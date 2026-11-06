"""Offline: measured scene -> is there a grasp? -> plan a push -> simulated replay (unfold, push, parking).
usage: nudge_probe.py scan.json frame.jpg out_dir"""
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
from box_extract.nudge import NudgePlanner
from box_extract.driver import BoxArmDriver, read_command_sag
from box_extract.opening_fallback import estimate_opening
import extract_from_box as cli

scan = json.loads(Path(sys.argv[1]).read_text()); frame = cv2.imread(sys.argv[2]); out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
if not any(d['label'] == 'box_opening' for d in scan['detections']):
    opening, _ = estimate_opening(frame, scan); scan['detections'].append(opening)
config = load_config(ROOT / 'config/default.json'); cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config.robot['calibration'], table_z_m=0., offsets=config.robot['kinematic_offsets_deg'])
params = SceneParameters(wall_clearance_m=.003, flap_planning_margin_m=.005, planning_wall_clearance_m=.004)
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='offline', ids=IdentityTracker())
ready = cli.choose_ready(backend, replace(scene, captured_at=time.time()))
planner = NudgePlanner(backend); planner.command_sag = read_command_sag(config, scan['robot_joint_deg'], out)
view = planning_view(replace(scene, placement=patches[0], captured_at=time.time()), params)
ids = [k for k in info['inside_box'] if not view.cubes[k].evidence.get('unmeasured')]
print('inside', ids, 'graspable', planner.graspable(view, ids, ready))
t = time.time(); plan, desc = planner.plan_nudge(view, ready, ids, print); print('nudge', desc, 'secs', round(time.time() - t))
print('phases', [(p.name, len(p.joints)) for p in plan.phases])
driver = BoxArmDriver(config, backend, out)
t = time.time(); rep = driver.preflight_nudge(plan, ExtractionCollisionWorld(backend, scene), ready, initial_joints=scan['robot_joint_deg'])
print('PREFLIGHT', rep, 'secs', round(time.time() - t))
