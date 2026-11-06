"""Is every planned waypoint accepted by the (tighter) guard world?  Plan vs guard consistency."""
import json, sys, time, os
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
import extract_from_box as cli
scan = json.load(open(sys.argv[1])); frame = cv2.imread(sys.argv[2])
config = load_config(ROOT / 'config/default.json'); cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config.robot['calibration'], table_z_m=0., offsets=config.robot['kinematic_offsets_deg'])
params = SceneParameters(wall_clearance_m=.003, flap_planning_margin_m=float(os.environ.get('MARGIN', .03)), planning_wall_clearance_m=float(os.environ.get('PCLR', .004)))
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='c', ids=IdentityTracker())
ready = cli.choose_ready(backend, replace(scene, captured_at=time.time()))
planner = TunedExtractionPlanner(backend)
plan, trial = cli.plan_scene(planner, scene, patches, ready, print, params)
guard_world = ExtractionCollisionWorld(backend, guard_view(trial))
for phase in plan.phases:
    bad = []
    for q in phase.joints:
        r = guard_world.reason(q, target=plan.target_id, jaw_contact=phase.jaw_contact, payload_local=plan.payload_local if phase.carrying else None,
                               support_contact=phase.support_contact, placed_cube=phase.placed_cube)
        if r: bad.append(r)
    print(phase.name, len(phase.joints), 'violations vs guard:', len(bad), sorted(set(bad))[:3])
