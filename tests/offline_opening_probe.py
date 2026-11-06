"""Run the rim-edge opening fallback on a saved scan and draw it (no server, no motors)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from box_extract.opening_fallback import estimate_opening

scan = json.load(open(sys.argv[1]))
frame = cv2.imread(scan['artifacts'] + '/frame.jpg')
detection, evidence = estimate_opening(frame, scan)
print(json.dumps(evidence, indent=1))
if detection:
    cv2.polylines(frame, [np.array(detection['silhouette_polygons_px'][0], np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    cv2.imwrite(sys.argv[2], frame)
    print('wrote', sys.argv[2], 'polygon', detection['silhouette_polygons_px'][0], 'area', detection['area_px'])
