"""Lightweight colour regions, quadrilateral faces and click segmentation."""
import cv2
import numpy as np

RANGES = {
    'red': [((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (179, 255, 255))],
    'yellow': [((17, 75, 70), (38, 255, 255))],
    'green': [((38, 60, 35), (90, 255, 255))],
    'blue': [((90, 85, 35), (135, 255, 255))],
}


def color_masks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = {}
    for color, ranges in RANGES.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))
        masks[color] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return masks


def detect_cubes(frame):
    detections = []
    for color, mask in color_masks(frame.image).items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            ratio = min(w, h) / max(w, h, 1.)
            solidity = area / max(1., cv2.contourArea(cv2.convexHull(contour)))
            if 450 < area < frame.image.shape[0]*frame.image.shape[1]*.15 and ratio > .38 and solidity > .86:
                hull = cv2.convexHull(contour)
                quad = cv2.approxPolyDP(hull, .025*cv2.arcLength(hull, True), True)
                if len(quad) == 4:
                    object_mask = np.zeros(mask.shape, np.uint8)
                    cv2.drawContours(object_mask, [contour], -1, 255, -1)
                    detections.append({'color': color, 'mask': object_mask, 'quad': quad.reshape(4, 2),
                        'center_px': np.mean(quad.reshape(4, 2), axis=0).tolist(),
                        'confidence': float(min(solidity, ratio+.25)), 'area_px': area})
    return sorted(detections, key=lambda item: item['center_px'][0])


def classify_color(frame, mask):
    masks = color_masks(frame.image)
    scores = {color: np.count_nonzero((candidate > 0) & (mask > 0)) for color, candidate in masks.items()}
    color = max(scores, key=scores.get)
    return color if scores[color] / max(1, np.count_nonzero(mask)) > .5 else 'unknown'


def detect_boxes(frame):
    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        quad = cv2.approxPolyDP(contour, .018*cv2.arcLength(contour, True), True)
        if 9000 < area < frame.image.size/3*.35 and len(quad) == 4 and cv2.isContourConvex(quad):
            points = quad.reshape(4, 2)
            x, y, w, h = cv2.boundingRect(quad)
            if x > 2 and y > 2 and x+w < gray.shape[1]-2 and y+h < gray.shape[0]-2 and min(w,h)/max(w,h) > .3:
                candidates.append({'polygon': points.tolist(), 'center_px': points.mean(0).tolist(), 'area_px': area,
                                   'opening_verified': False, 'height_verified': False})
    result = []
    for candidate in sorted(candidates, key=lambda item: -item['area_px']):
        if not any(np.linalg.norm(np.subtract(candidate['center_px'], other['center_px'])) < 40 for other in result):
            result.append(candidate)
    if len(result) < 2:
        result = detect_opening_edges(frame)
    return sorted(result, key=lambda item: item['center_px'][0])


def segment_clicked_object(frame, pixel, detections):
    x, y = map(int, pixel)
    if not (0 <= y < frame.image.shape[0] and 0 <= x < frame.image.shape[1]):
        raise ValueError('pixel is outside frame')
    matches = [d for d in detections if d['mask'][y, x] > 0]
    if len(matches) != 1:
        raise ValueError('click must select one supported coloured cube; unknown shapes need qualified segmentation')
    return matches[0]


def detect_opening_edges(frame):
    """Propose four supported rim lines; proposals never qualify box depth."""
    from itertools import combinations
    from box_extract.opening_fallback import cluster_lines, intersect
    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    detector = cv2.createLineSegmentDetector()
    lines = detector.detect(gray)[0]
    if lines is None:
        return []
    segments = lines.reshape(-1, 4)
    hough = cv2.HoughLinesP(cv2.Canny(gray,20,60),1,np.pi/360,55,minLineLength=80,maxLineGap=20)
    if hough is not None:
        segments = np.vstack((segments,hough.reshape(-1,4)))
    lengths = np.linalg.norm(segments[:, :2]-segments[:, 2:], axis=1)
    segments = segments[lengths > 20]
    groups = [sorted(cluster_lines(segments, vertical), key=lambda c: -c['length'])[:22] for vertical in (True, False)]
    proposed = []

    def coverage(line, a, b):
        points = a + np.linspace(0, 1, 30)[:, None]*(b-a)
        coordinate = (points-line['p']) @ line['d']
        covered = np.zeros(30, bool)
        for lo, hi in line['intervals']:
            covered |= (coordinate >= lo-12) & (coordinate <= hi+12)
        return float(covered.mean())

    for left, right in combinations(groups[0], 2):
        separation = abs(np.cross(left['d'], right['p']-left['p']))
        if 90 < separation < 420 and abs(left['d'] @ right['d']) > .995:
            for top, bottom in combinations(groups[1], 2):
                vertical = abs(np.cross(top['d'], bottom['p']-top['p']))
                if 90 < vertical < 430 and abs(top['d'] @ bottom['d']) > .995:
                    points = [intersect(left, top), intersect(right, top), intersect(right, bottom), intersect(left, bottom)]
                    if all(p is not None for p in points):
                        quad = np.array(points)
                        inside = np.all(quad > [3, 3]) and np.all(quad < [gray.shape[1]-3, gray.shape[0]-3])
                        if inside and cv2.isContourConvex(quad.astype(np.float32)):
                            scores = [coverage(line, quad[a], quad[b]) for line, a, b in ((top,0,1),(right,1,2),(bottom,2,3),(left,3,0))]
                            area = abs(cv2.contourArea(quad.astype(np.float32)))
                            if min(scores) >= .6 and 10000 < area < 140000:
                                proposed.append({'polygon': quad.tolist(), 'center_px': quad.mean(0).tolist(), 'area_px': area,
                                    'line_support': min(scores), 'opening_verified': False, 'height_verified': False})
    selected = []
    for item in sorted(proposed, key=lambda x: (-x['line_support'], -x['area_px'])):
        if not any(np.linalg.norm(np.subtract(item['center_px'], old['center_px'])) < 100 for old in selected):
            selected.append(item)
    return selected


def frame_quality(frame, minimum_sharpness=40.):
    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray,cv2.CV_64F).var())
    p95 = float(np.percentile(gray,95))
    return {'sharpness':sharpness,'brightness_p95':p95,
            'usable':sharpness >= minimum_sharpness and p95 >= 45.,
            'reason':'недостаточно света' if p95 < 45. else 'размытый кадр' if sharpness < minimum_sharpness else ''}
