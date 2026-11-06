"""Box opening from the rim edges, for scenes where SAM 3 returns no interior mask.

A white box on a white table with an unfolded lid is one silhouette for SAM 3, and its
interior prompts return nothing. The rim of the box body is still visible: its sides are
straight edges (the side along the hinge where the lid joins, and the sides against the table
as silhouette contour). The opening is the quadrilateral of four such lines that lies inside
the silhouette, contains the cubes, and has the best edge support. Each side is its own line:
under perspective and a rotated box the sides lean by different angles.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

MIN_SIDE_PX = 100
MIN_SUPPORT = .6
GOOD_SUPPORT = .65
MIN_INSIDE = .96
ANGLE_TOLERANCE_DEG = 3.5
OFFSET_TOLERANCE_PX = 7.
GAP_PX = 15.
INSET_PX = 6.
BORDER_MARGIN_PX = 15.
BOUNDARY_TOLERANCE_PX = 6.


def fill(polygons, shape):
    mask = np.zeros(shape, np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], 255)
    return mask > 0


def lsd_segments(frame, box_xyxy, minimum):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    lines = lines.reshape(-1, 4)
    x1, y1, x2, y2 = box_xyxy
    inside = ((lines[:, [0, 2]] >= x1 - 5) & (lines[:, [0, 2]] <= x2 + 5) & (lines[:, [1, 3]] >= y1 - 5) & (lines[:, [1, 3]] <= y2 + 5)).all(axis=1)
    lines = lines[inside]
    return lines[np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]) >= minimum]


def contour_segments(polygons, minimum):
    out = []
    for polygon in polygons:
        approx = cv2.approxPolyDP(np.asarray(polygon, np.int32).reshape(-1, 1, 2), 4., True).reshape(-1, 2).astype(float)
        for a, b in zip(approx, np.roll(approx, -1, axis=0)):
            if np.linalg.norm(b - a) >= minimum:
                out.append([*a, *b])
    return np.asarray(out, float).reshape(-1, 4)


def segment_angle(segments, vertical):
    """Angle in degrees: near 90 for vertical-ish lines, near 0 for horizontal-ish ones."""
    angle = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0])) % 180.
    return angle if vertical else np.where(angle > 90., angle - 180., angle)


def fit_line(segments):
    points = np.concatenate([segments[:, :2], segments[:, 2:]]).astype(np.float32)
    line = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).ravel()
    return line[:2] / np.linalg.norm(line[:2]), line[2:]


def cluster_lines(segments, vertical):
    """Merge near-collinear segments into lines with the intervals they cover."""
    if not len(segments):
        return []
    angles = segment_angle(segments, vertical)
    keep = np.abs(angles - 90.) < 35. if vertical else np.abs(angles) < 35.
    segments, angles = segments[keep], angles[keep]
    order = np.argsort(-np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]))
    clusters = []
    for i in order:
        segment = segments[i]
        for cluster in clusters:
            if abs(angles[i] - cluster['angle']) > ANGLE_TOLERANCE_DEG:
                continue
            normal = np.array([-cluster['d'][1], cluster['d'][0]])
            if max(abs((segment[:2] - cluster['p']) @ normal), abs((segment[2:] - cluster['p']) @ normal)) <= OFFSET_TOLERANCE_PX:
                cluster['segments'].append(segment)
                cluster['d'], cluster['p'] = fit_line(np.asarray(cluster['segments']))
                cluster['angle'] = float(segment_angle(np.array([[0., 0., *cluster['d']]]), vertical)[0])
                break
        else:
            d = (segment[2:] - segment[:2]) / np.linalg.norm(segment[2:] - segment[:2])
            clusters.append({'p': (segment[:2] + segment[2:]) / 2., 'd': d, 'angle': float(angles[i]), 'segments': [segment]})
    for cluster in clusters:
        spans = sorted(sorted(((s[:2] - cluster['p']) @ cluster['d'], (s[2:] - cluster['p']) @ cluster['d']))
                       for s in cluster['segments'])
        merged = []
        for lo, hi in spans:
            if merged and lo - merged[-1][1] <= GAP_PX:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        cluster['intervals'] = merged
        cluster['length'] = float(sum(hi - lo for lo, hi in merged))
    return [c for c in clusters if c['length'] >= 80.]


def intersect(a, b):
    matrix = np.column_stack((a['d'], -b['d']))
    if abs(np.linalg.det(matrix)) < 1e-3:
        return None
    t, _ = np.linalg.solve(matrix, b['p'] - a['p'])
    return a['p'] + t * a['d']


def side_support(cluster, start, end, boundary_distance):
    """Fraction of the side covered by the line's own segments or lying on the silhouette boundary."""
    count = max(2, int(np.linalg.norm(end - start) / 4.))
    points = start + np.linspace(0., 1., count)[:, None] * (end - start)
    along = (points - cluster['p']) @ cluster['d']
    covered = np.zeros(count, bool)
    for lo, hi in cluster['intervals']:
        covered |= (along >= lo) & (along <= hi)
    xs, ys = np.rint(points[:, 0]).astype(int), np.rint(points[:, 1]).astype(int)
    valid = (xs >= 0) & (xs < boundary_distance.shape[1]) & (ys >= 0) & (ys < boundary_distance.shape[0])
    on_boundary = np.zeros(count, bool)
    on_boundary[valid] = boundary_distance[ys[valid], xs[valid]] <= BOUNDARY_TOLERANCE_PX
    return float(np.mean(covered | on_boundary))


def offset_line(cluster, centroid, distance):
    normal = np.array([-cluster['d'][1], cluster['d'][0]])
    if (centroid - cluster['p']) @ normal < 0:
        normal = -normal
    return {'p': cluster['p'] + normal * distance, 'd': cluster['d']}


def crossing(line, centre, axis):
    """Coordinate along `axis` where the line crosses the box centre in the other coordinate."""
    other = 1 - axis
    if abs(line['d'][other]) < 1e-6:
        return None
    t = (centre[other] - line['p'][other]) / line['d'][other]
    return float((line['p'] + t * line['d'])[axis])


def polygon_boundary(polygons, shape):
    """Silhouette outline pixels, without the cut where the silhouette meets the image border."""
    outline = np.zeros(shape, np.uint8)
    for polygon in polygons:
        cv2.polylines(outline, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], True, 1, 1)
    inner = np.zeros(shape, np.uint8)
    inner[8:-8, 8:-8] = 1
    return (outline & inner).astype(bool)


class OpeningTracker:
    """Steady opening across scans: the mean of the last scans' corners (a bimodal contour edge would flip a median),
    scans far from the median rejected as outliers, and the result held fixed until the mean has moved by `relock_px`.
    The project's scene checks compare the opening between scans to a few millimetres; a re-measured contour never
    repeats that closely, a held one does.
    """
    outlier_px = 30.
    relock_px = 8.

    def __init__(self, size=15, lock_after=6):
        self.size, self.lock_after, self.history, self.center, self.locked = size, lock_after, [], None, None

    def smooth(self, corners, center):
        quad = np.asarray(corners, float)
        if self.center is not None and np.linalg.norm(np.asarray(center, float) - self.center) > 40.:
            self.history, self.locked = [], None                  # the box was moved: start over
        self.center = np.asarray(center, float)
        if len(self.history) >= 3 and np.abs(quad - np.median(self.history, axis=0)).max() > self.outlier_px:
            return self.locked if self.locked is not None else np.median(self.history, axis=0)
        self.history = (self.history + [quad])[-self.size:]
        mean = np.mean(self.history, axis=0)
        if len(self.history) >= self.lock_after and (self.locked is None or np.abs(mean - self.locked).max() > self.relock_px):
            self.locked = mean
        return self.locked if self.locked is not None else mean


MEMORY = Path(__file__).resolve().parents[1] / 'calibration' / 'last_opening.json'
MEMORY_MAX_AGE_S = 6 * 3600


def remember(tracker, box_center):
    """The held opening survives the process, for scenes where the arm hides the rim (the arm above the box)."""
    if tracker.locked is not None:
        try:
            MEMORY.write_text(json.dumps({'quad': np.asarray(tracker.locked).tolist(), 'box_center_px': [float(v) for v in box_center], 'time': time.time()}))
        except OSError:
            pass


def recall(box_center, max_distance_px=30.):
    try:
        data = json.loads(MEMORY.read_text())
    except (OSError, ValueError):
        return None
    if time.time() - data['time'] > MEMORY_MAX_AGE_S or np.linalg.norm(np.asarray(data['box_center_px']) - np.asarray(box_center, float)) > max_distance_px:
        return None
    return np.asarray(data['quad'], float)


TRACKER = OpeningTracker()


def _detection(corners, box, shape, source):
    polygon = [[int(round(x)), int(round(y))] for x, y in corners]
    mask_out = fill([polygon], shape)
    ys, xs = np.nonzero(mask_out)
    angle = math.degrees(math.atan2(corners[1][1] - corners[0][1], corners[1][0] - corners[0][0]))
    return {'label': 'box_opening', 'score': .8 * min(1., float(box['score'])), 'source': source,
            'center_px': [float(xs.mean()), float(ys.mean())], 'area_px': float(mask_out.sum()),
            'box_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], 'angle_deg': angle,
            'silhouette_polygons_px': [polygon],
            'drop_geometry': {'source': 'rim-edge fallback; measured by the extraction program, not the server'}}


def _estimate_opening(frame, scan, *, inset_px=INSET_PX, tracker=None):
    """Return a `box_opening` detection (public dict) or None, plus an evidence dict."""
    boxes = [d for d in scan['detections'] if d['label'] == 'box']
    if len(boxes) != 1:
        return None, {'reason': f'{len(boxes)} boxes'}
    cubes = [d for d in scan['detections'] if d['label'] == 'cube' and d.get('silhouette_polygons_px')]
    shape = frame.shape[:2]
    polygons = boxes[0]['silhouette_polygons_px']
    mask = fill(polygons, shape)
    for cube in cubes:
        mask |= fill(cube.get('silhouette_polygons_px', []), shape) & cv2.dilate(mask.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
    allowed = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    boundary = polygon_boundary(polygons, shape)
    boundary_distance = cv2.distanceTransform(np.uint8(~boundary), cv2.DIST_L2, 3)
    x1, y1, x2, y2 = boxes[0]['box_xyxy']
    minimum = .12 * min(x2 - x1, y2 - y1)
    segments = np.vstack([lsd_segments(frame, boxes[0]['box_xyxy'], minimum), contour_segments(polygons, minimum)])
    verticals, horizontals = cluster_lines(segments, True), cluster_lines(segments, False)
    if len(verticals) < 2 or len(horizontals) < 2:
        return None, {'reason': 'too few straight edges', 'verticals': len(verticals), 'horizontals': len(horizontals)}
    centre = np.array(boxes[0]['center_px'], float)
    cube_centers = np.array([c['center_px'] for c in cubes], float).reshape(-1, 2)
    ordered_v = sorted(((crossing(l, centre, 0), l) for l in verticals if crossing(l, centre, 0) is not None), key=lambda item: item[0])
    ordered_h = sorted(((crossing(l, centre, 1), l) for l in horizontals if crossing(l, centre, 1) is not None), key=lambda item: item[0])
    candidates = []
    for i, (xl, left) in enumerate(ordered_v):
        for xr, right in ordered_v[i + 1:]:
            if xr - xl < MIN_SIDE_PX:
                continue
            for j, (yt, top) in enumerate(ordered_h):
                for yb, bottom in ordered_h[j + 1:]:
                    if yb - yt < MIN_SIDE_PX:
                        continue
                    corners = [intersect(top, left), intersect(top, right), intersect(bottom, right), intersect(bottom, left)]
                    if any(c is None for c in corners):
                        continue
                    quad = np.array(corners)
                    # A rim cut off by the image border is unreliable (an unfolded lid reaching the edge looks like one).
                    if not np.all((quad > BORDER_MARGIN_PX) & (quad < [shape[1] - BORDER_MARGIN_PX, shape[0] - BORDER_MARGIN_PX])):
                        continue
                    sides = [float(np.linalg.norm(quad[(k + 1) % 4] - quad[k])) for k in range(4)]
                    if min(sides) < MIN_SIDE_PX or not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
                        continue
                    area = float(cv2.contourArea(quad.astype(np.float32)))
                    if area > .9 * mask.sum() or area < 8000:
                        continue
                    supports = (side_support(top, quad[0], quad[1], boundary_distance), side_support(right, quad[1], quad[2], boundary_distance),
                                side_support(bottom, quad[3], quad[2], boundary_distance), side_support(left, quad[0], quad[3], boundary_distance))
                    if min(supports) < MIN_SUPPORT:
                        continue
                    held = int(sum(cv2.pointPolygonTest(quad.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) > 0 for x, y in cube_centers))
                    length = sum(s * l for s, l in zip(supports, sides))
                    # Cubes held first; then the largest quadrilateral whose four sides are all well supported (the rim,
                    # not the floor edge inside it); otherwise the best total support.
                    score = held * 1e9 + (1e8 + area if min(supports) >= GOOD_SUPPORT else 0.) + length
                    candidates.append((score, quad, (top, right, bottom, left), supports, held, area))
    candidates.sort(key=lambda item: -item[0])
    for score, quad, lines, supports, held, area in candidates[:60]:
        inside = np.zeros(shape, np.uint8)
        cv2.fillPoly(inside, [np.rint(quad).astype(np.int32).reshape(-1, 1, 2)], 1)
        if np.count_nonzero(inside.astype(bool) & allowed) < MIN_INSIDE * np.count_nonzero(inside):
            continue
        centroid = quad.mean(0)
        top, right, bottom, left = (offset_line(line, centroid, inset_px) for line in lines)
        corners = [intersect(top, left), intersect(top, right), intersect(bottom, right), intersect(bottom, left)]
        if any(c is None for c in corners):
            continue
        corners = (TRACKER if tracker is None else tracker).smooth(corners, boxes[0]['center_px'])
        remember(TRACKER if tracker is None else tracker, boxes[0]['center_px'])
        detection = _detection(corners, boxes[0], shape, 'rim-edge fallback')
        polygon = detection['silhouette_polygons_px'][0]
        evidence = {'corners_px': polygon, 'support': [float(v) for v in supports], 'cubes_inside': held, 'area_px': float(area),
                    'candidates': len(candidates),
                    'runner_up': [{'held': c[4], 'support': [round(float(v), 2) for v in c[3]], 'corners': np.rint(c[1]).astype(int).tolist()} for c in candidates[:6]]}
        return detection, evidence
    remembered = recall(boxes[0]['center_px'])
    if remembered is not None:
        return _detection(remembered, boxes[0], shape, 'remembered (the rim is hidden)'), {'corners_px': np.rint(remembered).astype(int).tolist(), 'remembered': True}
    return None, {'reason': 'no quadrilateral with four supported sides inside the silhouette',
                  'verticals': len(verticals), 'horizontals': len(horizontals), 'candidates': len(candidates)}


def estimate_opening(frame, scan, *, inset_px=INSET_PX, tracker=None):
    """Return a `box_opening` detection (public dict) or None, plus an evidence dict.

    The measurement first; when the rim cannot be measured (the arm hides it) and a recent opening of the same box
    was held, that one.
    """
    detection, evidence = _estimate_opening(frame, scan, inset_px=inset_px, tracker=tracker)
    if detection is None:
        boxes = [d for d in scan['detections'] if d['label'] == 'box']
        remembered = recall(boxes[0]['center_px']) if len(boxes) == 1 else None
        if remembered is not None:
            return _detection(remembered, boxes[0], frame.shape[:2], 'remembered (the rim is hidden)'), {'corners_px': np.rint(remembered).astype(int).tolist(), 'remembered': True}
    return detection, evidence
