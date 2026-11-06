"""Give the project's own tools the box opening this program measures when the server finds none.

The server's /api/scan-hybrid returns a `box_opening` only when its interior prompts or its
edge-pair search succeed; a box with an unfolded lid defeats both. `complete_scan_record`
adds the rim-edge opening (box_extract.opening_fallback) in the same format the server
produces, so the project's calibration and planning scripts can run unchanged.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .opening_fallback import estimate_opening, fill

PROJECT = Path.home() / 'Projects/so101_autonomous_pick_place'


def complete_scan_record(record, *, log=None):
    from so101_pick_place.geometry import TableProjection
    from so101_pick_place.geometry.grasp_geometry import estimate_box_drop_geometry
    from so101_pick_place.vision.mask_geometry import detection_from_mask

    detections = record.get('detections', [])
    if any(d['label'] == 'box_opening' and d.get('drop_geometry') for d in detections):
        return record
    frame = cv2.imread(str(Path(record['artifacts']) / 'frame.jpg'))
    if frame is None:
        return record
    opening, evidence = estimate_opening(frame, record)
    if opening is None:
        if log:
            log(f'rim-edge fallback found no opening: {evidence}')
        return record
    mask = np.uint8(fill(opening['silhouette_polygons_px'], frame.shape[:2]) * 255)
    detection = detection_from_mask('box_opening', opening['score'], mask > 0)
    if detection is None:
        return record
    calibration = json.loads((PROJECT / 'config/calibration.json').read_text())
    heights = json.loads((PROJECT / 'config/default.json').read_text()).get('scene', {}).get('box_rim_height_interval_m', [.06, .14])
    item = detection.public()
    contours, _ = cv2.findContours(detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    item['silhouette_polygons_px'] = [cv2.approxPolyDP(c, .5, True).reshape(-1, 2).tolist() for c in contours]
    point = TableProjection(np.asarray(calibration['camera_matrix']), np.asarray(calibration.get('distortion', [0, 0, 0, 0, 0])),
                            np.asarray(calibration['camera_from_robot']), float(calibration.get('table_z_m', 0.)),
                            float(calibration['reprojection_rmse_px'])).pixel_to_table(detection.center_px)
    item['position_robot_m'] = list(point.xyz_m)
    item['uncertainty_m'] = point.uncertainty_m
    drop = estimate_box_drop_geometry(detection, calibration, rim_height_interval_m=tuple(heights))
    item['drop_geometry'] = drop.public() | {'height_source': 'operating bound, not a height measurement'}
    item['source'] = 'rim-edge fallback (box_extract.opening_fallback)'
    item['fallback_evidence'] = evidence
    polygon = np.asarray(item['silhouette_polygons_px'][0], np.float32).reshape(-1, 1, 2)
    for d in detections:
        if d['label'] == 'cube':
            d['inside_box'] = bool(cv2.pointPolygonTest(polygon, (float(d['center_px'][0]), float(d['center_px'][1])), False) > 0)
    record['detections'] = [*detections, item]
    return record
