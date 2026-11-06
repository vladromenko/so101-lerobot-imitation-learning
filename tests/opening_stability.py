"""Rim-edge opening on N consecutive live scans: success rate and spread (no motion)."""
import json, sys, urllib.request
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from box_extract.opening_fallback import estimate_opening
n = int(sys.argv[1]); rows = []
for i in range(n):
    rec = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8010/api/scan-hybrid', method='POST', data=b''), timeout=120))
    frame = cv2.imread(rec['artifacts'] + '/frame.jpg')
    json.dump(rec, open(f'tmp/stab_{i}.json', 'w'))
    opening, ev = estimate_opening(frame, rec)
    if opening is None:
        print(i, 'FAILED', ev); continue
    rows.append(np.array(opening['silhouette_polygons_px'][0], float)); print(i, 'ok', ev['corners_px'], [round(v, 2) for v in ev['support']])
if rows:
    a = np.array(rows); print('success', len(rows), '/', n, 'corner spread px (max over corners)', np.round((a.max(0) - a.min(0)).max(), 1))
