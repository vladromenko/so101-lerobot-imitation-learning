import json, math, sys, cv2, numpy as np
rec = json.load(open(sys.argv[1]))
frame = cv2.imread(rec["artifacts"] + "/frame.jpg")
box = next(d for d in rec["detections"] if d["label"] == "box")
mask = np.zeros(frame.shape[:2], np.uint8)
for p in box["silhouette_polygons_px"]:
    cv2.fillPoly(mask, [np.array(p, np.int32)], 255)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print("box mask area", int((mask>0).sum()), "bbox", box["box_xyxy"])
for name, (x1, y1, x2, y2) in {"lid": (30, 130, 280, 450), "floor": (350, 200, 530, 470), "left_wall_strip": (305, 200, 335, 450), "top_wall_strip": (350, 170, 530, 188), "table": (600, 200, 700, 400)}.items():
    r = gray[y1:y2, x1:x2]
    print(f"{name:16s} mean {r.mean():6.1f} std {r.std():5.1f} p10 {np.percentile(r,10):5.1f} p90 {np.percentile(r,90):5.1f}")
print("box p75", np.percentile(gray[mask > 0], 75))
lines = cv2.createLineSegmentDetector().detect(gray)[0]
segs = []
for l in lines:
    a, b = l.reshape(2, 2); n = float(np.linalg.norm(b - a))
    if n > 80 and a.min() >= -5 and max(a[0], b[0]) <= 560 and max(a[1], b[1]) <= 500:
        ang = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])) % 180
        segs.append((round(n), round(ang), [round(v) for v in (*a, *b)]))
for s in sorted(segs, reverse=True)[:40]: print(s)
