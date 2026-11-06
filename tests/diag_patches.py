"""Why are there no free table patches? Counts of candidates removed by each condition, on a live scan."""
import json, sys, urllib.request
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'; sys.path.insert(0, str(ROOT / 'src'))
from so101_pick_place.config import load_config
from so101_pick_place.kinematics import So101ModelPlanner
from box_extract.scene_builder import *
from box_extract.server_compat import complete_scan_record
config = load_config(ROOT / 'config/default.json'); cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config.robot['calibration'], table_z_m=0., offsets=config.robot['kinematic_offsets_deg'])
params = SceneParameters(wall_clearance_m=.003, patch_margin_m=.025, patch_reach_m=(.12, .31))
scans, frames = [], []
for _ in range(3):
    r = complete_scan_record(json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8010/api/scan-hybrid', method='POST', data=b''), timeout=120)))
    scans.append(r); frames.append(cv2.imread(r['artifacts'] + '/frame.jpg'))
scene, patches, info = build_scene(scans, frames, cal, backend, params, revision='d', ids=IdentityTracker())
print('patches', len(patches), 'cubes', {k: np.round(c.center_m[:2], 3).tolist() for k, c in scene.cubes.items()}, 'unknown', len(info['unknown']))
shape = frames[0].shape; proj = projection(cal, 0.); half = params.patch_side_m / 2; base = np.linalg.inv(scene.box.base_from_box)
masks = []
for scan, frame in zip(scans, frames):
    occ = robot_mask(backend, scan.get('robot_joint_deg'), cal, shape)
    for d in scan['detections']:
        if d.get('label') in {'cube', 'box', 'box_opening', 'background_surface'}:
            g = np.zeros(shape[:2], np.uint8)
            cv2.fillPoly(g, [np.asarray(p, np.int32) for p in d.get('silhouette_polygons_px', [])], 255)
            occ |= cv2.dilate(g, np.ones((15, 15), np.uint8))
    masks.append((occ, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
counts = {'reach': 0, 'near_box': 0, 'near_cube': 0, 'image_edge': 0, 'occupied': 0, 'dark': 0, 'std': 0, 'ok': 0}
for x in np.arange(.10, .34, .02):
    for y in np.arange(-.32, .24, .02):
        xy = np.array([x, y]); r = float(np.hypot(x, y))
        if not params.patch_reach_m[0] <= r <= params.patch_reach_m[1]: counts['reach'] += 1; continue
        local = base[:2, :2] @ xy + base[:2, 3]
        lo = np.asarray(scene.box.evidence['silhouette_min_xy']) - params.patch_box_clearance_m - half; hi = np.asarray(scene.box.evidence['silhouette_max_xy']) + params.patch_box_clearance_m + half
        if np.all(local > lo) and np.all(local < hi): counts['near_box'] += 1; continue
        if any(np.linalg.norm(c.center_m[:2] - xy) < half + params.patch_margin_m + c.side_m for c in scene.cubes.values()): counts['near_cube'] += 1; continue
        margin = half + params.patch_margin_m
        px = np.array([proj.table_to_pixel((xy[0] + sx * margin, xy[1] + sy * margin, 0.)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        if np.any(px < 25) or np.any(px[:, 0] > shape[1] - 25) or np.any(px[:, 1] > shape[0] - 25): counts['image_edge'] += 1; continue
        region = np.zeros(shape[:2], np.uint8); cv2.fillConvexPoly(region, np.rint(px).astype(np.int32), 255)
        bad = None
        for occ, gray in masks:
            if np.any((region > 0) & (occ > 0)): bad = 'occupied'; break
            v = gray[region > 0]
            if np.percentile(v, 5) < 110: bad = 'dark'; break
            if v.std() > 32: bad = 'std'; break
        counts[bad or 'ok'] += 1
print(counts)
