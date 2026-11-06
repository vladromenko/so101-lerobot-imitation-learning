"""Why is the expected quadrilateral not chosen? Print supports of the candidate lines near given corners."""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import box_extract.opening_fallback as of
scan = json.load(open(sys.argv[1])); frame = cv2.imread(scan['artifacts'] + '/frame.jpg')
box = next(d for d in scan['detections'] if d['label'] == 'box')
polys = box['silhouette_polygons_px']
segs = np.vstack([of.lsd_segments(frame, box['box_xyxy'], .12 * min(box['box_xyxy'][2] - box['box_xyxy'][0], box['box_xyxy'][3] - box['box_xyxy'][1])), of.contour_segments(polys, 40)])
V, H = of.cluster_lines(segs, True), of.cluster_lines(segs, False)
print('V'); [print('  ', np.round(c['p']), np.round(c['d'], 3), round(c['length']), [[round(a), round(b)] for a, b in c['intervals']]) for c in V]
print('H'); [print('  ', np.round(c['p']), np.round(c['d'], 3), round(c['length']), [[round(a), round(b)] for a, b in c['intervals']]) for c in H]
