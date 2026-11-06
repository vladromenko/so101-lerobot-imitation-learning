"""Offline probe: build the scene from a saved scan + frame (no server, no motors)."""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'
from so101_pick_place.config import load_config
from so101_pick_place.kinematics import So101ModelPlanner
from box_extract.scene_builder import *

scan = json.loads(Path(sys.argv[1]).read_text()); frame = cv2.imread(sys.argv[2])
config = json.loads((ROOT / 'config/default.json').read_text())
cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config['robot']['calibration'], table_z_m=cal['table_z_m'], offsets=config['robot']['kinematic_offsets_deg'])
params = SceneParameters()
t = time.time()
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='probe', ids=IdentityTracker())
print('built in', round(time.time() - t, 2), 's', info)
b = scene.box
print('box center', b.base_from_box[:2, 3], 'yaw deg', np.rad2deg(np.arctan2(b.base_from_box[1, 0], b.base_from_box[0, 0])))
print('inner', b.inner_min_xy, b.inner_max_xy, 'size mm', (b.inner_max_xy - b.inner_min_xy) * 1000)
print("wall", b.outer_min_xy, b.outer_max_xy, "flaps", b.evidence["flaps"])
for k, c in scene.cubes.items():
    print(k, 'center', np.round(c.center_m, 4), 'side mm', round(c.side_m * 1000, 1), 'inside', inside_box(c, b))
raw = {d['center_px'][0]: d for d in scan['detections'] if d['label'] == 'cube'}
for d in raw.values():
    top = np.array(d['geometry']['corners_top_m'])
    c = min(scene.cubes.values(), key=lambda q: np.linalg.norm(q.center_m[:2] - np.array(d['geometry']['center_volume_m'][:2])))
    err = max(min(np.linalg.norm(top[:, None, :2] - c.vertices_m[None, 4:, :2], axis=2).max(1)) for _ in [0])
    print('top-vertex consistency (mm):', round(1000 * max(np.linalg.norm(top[i, :2] - c.vertices_m[4:, :2], axis=1).min() for i in range(4)), 3))
print('patches', [np.round(p.evidence['center_xy_m'], 3).tolist() for p in patches])
scene.require_complete()
print('scene complete OK; coverage', scene.coverage_verified)
