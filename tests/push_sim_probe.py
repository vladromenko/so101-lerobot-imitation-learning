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
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld
from box_extract.driver import BoxArmDriver
box = [view.box.base_from_box[:2, k] / np.linalg.norm(view.box.base_from_box[:2, k]) for k in (0, 1)]
plan = planner.plan_push(view, ready, ids[0], np.r_[-box[0], 0.], .01)
print('PLAN', [(p.name, len(p.joints)) for p in plan.phases], flush=True)
bare = ExtractionCollisionWorld(backend, replace(scene, cubes={k: v for k, v in scene.cubes.items() if k != ids[0]}, placement=patches[0]))
nud = plan.phases[3].joints
from so101_pick_place.kinematics.so101_backend import ARM_JOINTS
for i in range(len(nud) - 1):
    a, b = nud[i], nud[i + 1]; span = max(abs(a[j] - b[j]) for j in ARM_JOINTS)
    for al in np.linspace(0, 1, max(2, int(np.ceil(span / .2)) + 1)):
        q = {j: a[j] + al * (b[j] - a[j]) for j in (*ARM_JOINTS, 'gripper')}
        for kind, qq in (('goal', q), ('command', planner._command_pose(q))):
            r = bare.reason(qq)
            if r:
                fr = backend.model.frames(qq); w = backend._transform_points(fr['gripper_link'], backend.meshes['gripper_link'])
                print('FAIL waypoint', i, 'al', round(al, 2), kind, r, 'gripper_link min z', round(float(w[:, 2].min()), 4)); break
        else: continue
        break
    else: continue
    break
driver = BoxArmDriver(config, backend, Path('/tmp/pushsim'))
_orig = BoxArmDriver.guard_tick
def _spy(self, tick):
    try:
        return _orig(self, tick)
    except RuntimeError as exc:
        from so101_pick_place.kinematics.so101_backend import ARM_JOINTS as AJ
        goal = tick['goal']
        dev, idx = min((max(abs(goal[j] - q[j]) for j in AJ), i) for i, q in enumerate(self.phase.joints))
        print('SPY', self.phase.name, exc, 'nearest waypoint', idx, 'of', len(self.phase.joints), 'dev', round(dev, 2), {j: round(goal[j], 1) for j in AJ}, 'waypoint', {j: round(self.phase.joints[idx][j], 1) for j in AJ})
        raise
BoxArmDriver.guard_tick = _spy
t = time.time(); print('PREFLIGHT', driver.preflight_nudge(plan, ExtractionCollisionWorld(backend, replace(scene, placement=patches[0])), ready, initial_joints=scan['robot_joint_deg']), round(time.time() - t), 's')

