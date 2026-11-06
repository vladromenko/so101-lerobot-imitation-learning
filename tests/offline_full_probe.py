"""Whole offline pipeline on a saved (or synthetic) scan: scene -> ready pose -> plan -> full simulated replay."""
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
from box_extract.driver import BoxArmDriver
import extract_from_box as cli

scan = json.loads(Path(sys.argv[1]).read_text()); frame = cv2.imread(sys.argv[2]); out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
config = load_config(ROOT / 'config/default.json'); cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config.robot['calibration'], table_z_m=0., offsets=config.robot['kinematic_offsets_deg'])
import os
params = SceneParameters(wall_clearance_m=.003, flap_planning_margin_m=float(os.environ.get('MARGIN', .03)), planning_wall_clearance_m=float(os.environ.get('PCLR', .005)))
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='offline', ids=IdentityTracker())
print('patches', len(patches), 'inside', info['inside_box'])
ready = cli.choose_ready(backend, replace(scene, captured_at=time.time())); print('ready', {k: round(v, 1) for k, v in ready.items()})
planner = TunedExtractionPlanner(backend)
t = time.time(); plan, trial = cli.plan_scene(planner, scene, patches, ready, print, params); print('plan secs', round(time.time() - t))
cli.write_overlay(out / 'overlay.jpg', frame, trial, plan, cal)
print('phases', [(p.name, len(p.joints)) for p in plan.phases])
driver = BoxArmDriver(config, backend, out)
t = time.time(); rep = driver.preflight_full(plan, ExtractionCollisionWorld(backend, trial), ready, initial_joints=scan['robot_joint_deg'], params=params)
print('PREFLIGHT', rep, 'secs', round(time.time() - t))
