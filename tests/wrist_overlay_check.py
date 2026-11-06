"""Independent check of a wrist-camera calibration candidate against the overview camera (no motion).

The overview camera measures the cubes in robot coordinates; the candidate's hand-eye and
intrinsics project those cubes into a wrist frame taken at the same joints. A good
calibration puts the wireframe on the cubes the wrist camera sees.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'
sys.path.insert(0, str(ROOT / 'src'))
from so101ik import Model
from so101_pick_place.config import load_config
from so101_pick_place.vision.wrist_camera_geometry import camera_projection

API = 'http://127.0.0.1:8010'
candidate_path, out = sys.argv[1], sys.argv[2]
candidate = json.load(open(candidate_path))
candidate = candidate.get('candidate', candidate)
config = load_config(ROOT / 'config/default.json')
cal = json.loads((ROOT / 'config/calibration.json').read_text())
model = Model(calibration_file=config.robot['calibration'], offsets=config.robot.get('kinematic_offsets_deg'))
model.table_z = float(cal['table_z_m'])

scan = json.load(urllib.request.urlopen(urllib.request.Request(API + '/api/scan-hybrid', method='POST', data=b''), timeout=120))
wrist_bytes = urllib.request.urlopen(API + '/snapshot/wrist.jpg', timeout=10).read()
wrist = cv2.imdecode(np.frombuffer(wrist_bytes, np.uint8), cv2.IMREAD_COLOR)
joints = scan['robot_joint_deg']
print('joints', {k: round(v, 1) for k, v in joints.items()})
projection = camera_projection(candidate, model, joints, calibration_control_only=True)
k, d = np.asarray(projection['camera_matrix']), np.asarray(projection['distortion'])
base_to_camera = np.asarray(projection['camera_from_robot'])
rvec, _ = cv2.Rodrigues(base_to_camera[:3, :3])
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
for det in scan['detections']:
    if det['label'] != 'cube' or not det.get('geometry'):
        continue
    top = np.asarray(det['geometry']['corners_top_m'], float)
    height = float(det['geometry']['height_m'])
    pts = np.vstack([top, top - [0, 0, height]])
    camera_pts = pts @ base_to_camera[:3, :3].T + base_to_camera[:3, 3]
    front = bool((camera_pts[:, 2] > .02).all())
    image, _ = cv2.projectPoints(pts, rvec, base_to_camera[:3, 3], k, d)
    image = image.reshape(-1, 2)
    inside = bool(((image[:, 0] > 0) & (image[:, 0] < wrist.shape[1]) & (image[:, 1] > 0) & (image[:, 1] < wrist.shape[0])).all())
    print('cube at robot', [round(v, 3) for v in det['position_robot_m'][:2]], 'in front', front, 'fully in view', inside,
          'centre px', image.mean(0).round(0).tolist())
    for a, b in edges:
        cv2.line(wrist, tuple(int(v) for v in image[a]), tuple(int(v) for v in image[b]), (0, 255, 0), 3)
cv2.imwrite(out, wrist)
