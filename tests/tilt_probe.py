"""Is the cube graspable where it lies if the tool may tilt more? usage: tilt_probe.py scan frame"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'scripts'))
ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'; sys.path.insert(0, str(ROOT / 'scripts'))
from so101_pick_place.config import load_config
from so101_pick_place.kinematics import So101ModelPlanner
from box_extract.scene_builder import *
from box_extract.nudge import NudgePlanner
from box_extract.driver import read_command_sag
from box_extract.opening_fallback import estimate_opening
import extract_from_box as cli
scan = json.loads(Path(sys.argv[1]).read_text()); frame = cv2.imread(sys.argv[2])
if not any(d['label'] == 'box_opening' for d in scan['detections']):
    opening, _ = estimate_opening(frame, scan); scan['detections'].append(opening)
config = load_config(ROOT / 'config/default.json'); cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config.robot['calibration'], table_z_m=0., offsets=config.robot['kinematic_offsets_deg'])
params = SceneParameters(wall_clearance_m=.003, flap_planning_margin_m=.005, planning_wall_clearance_m=.004)
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='offline', ids=IdentityTracker())
ready = cli.choose_ready(backend, replace(scene, captured_at=time.time()))
planner = NudgePlanner(backend); planner.command_sag = read_command_sag(config, scan['robot_joint_deg'], Path('/tmp/tilt'))
view = planning_view(replace(scene, placement=patches[0], captured_at=time.time()), params)
ids = [k for k in info['inside_box'] if not view.cubes[k].evidence.get('unmeasured')]
for pitches, down in (((-90., -70., -50.), .5), ((-90., -70., -50., -40., -30., -20.), .3), ((-60., -45., -35., -25., -15.), .2)):
    planner.grasp_pitches_deg, planner.min_tool_down = pitches, down
    for k in ids:
        t = time.time(); ok, tried, why = planner.entry_report(view, k, ready, limit=30)
        print(pitches, down, k, 'entries ok', ok, 'of', tried, dict(why.most_common(3)), round(time.time() - t), 's', flush=True)
