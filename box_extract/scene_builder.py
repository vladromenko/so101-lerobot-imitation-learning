"""Metric scene (box, cubes, free table patches) from overview SAM 3 scans.

Everything here is derived from fresh camera measurements and the installed
overview calibration; no saved joint angles or fixed task coordinates.  Where
the overview camera cannot measure something (cardboard floor thickness, cube
height above the floor, table edges) the assumption is explicit in
``SceneParameters`` and is copied into the evidence of the scene objects.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import cv2
import numpy as np

from so101_pick_place.experiments.box_extraction.scene import ExtractionScene, SupportPatch
from so101_pick_place.geometry.box_interior import MeasuredBoxInterior
from so101_pick_place.geometry.cuboid_vision import cube_vertices
from so101_pick_place.geometry.spatial_cube import SpatialCube
from so101_pick_place.geometry.table_projection import TableProjection


@dataclass(frozen=True)
class FlapAwareBox(MeasuredBoxInterior):
    """Box whose walls stop at the rim; flaps are separate, taller solids on the sides that have them."""
    flap_parts: tuple = ()
    flap_faces: tuple = ()         # (axis, sign) of each flap: the side that touches the box wall
    flap_inner_inflate_m: float = 0.   # growth toward the opening, limited to the assumed wall footprint
    flap_inflate_m: float = 0.     # >0 only for planning: keeps planned paths off the flaps by more than the driver's tracking error

    def solid_parts(self):
        grow, flaps = self.flap_inflate_m, []
        for (lo, hi, label), (axis, sign) in zip(self.flap_parts, self.flap_faces):
            lo, hi = np.asarray(lo, float).copy(), np.asarray(hi, float).copy()
            lo_grow, hi_grow = np.full(3, grow), np.full(3, grow)
            (hi_grow if sign < 0 else lo_grow)[axis] = self.flap_inner_inflate_m   # toward the opening: only inside the wall footprint
            flaps.append((lo - lo_grow, hi + hi_grow, label))
        return [*super().solid_parts(), *flaps]


def planning_view(scene, params):
    """Scene for the planner: flaps inflated away from the opening and a larger wall clearance.

    The execution guard (``guard_view``) keeps the true, tighter clearance, so the driver's
    tracking error is absorbed by the difference between what is planned and what is enforced.
    """
    box = replace(scene.box, flap_inflate_m=float(params.flap_planning_margin_m),
                  flap_inner_inflate_m=float(max(0., min(params.flap_planning_margin_m, params.wall_thickness_m - .001))),
                  clearance_m=max(float(params.planning_wall_clearance_m), float(scene.box.evidence['guard_clearance_m'])))
    return replace(scene, box=box)


def guard_view(scene):
    return replace(scene, box=replace(scene.box, flap_inflate_m=0., flap_inner_inflate_m=0., clearance_m=float(scene.box.evidence['guard_clearance_m'])))


class SceneError(ValueError):
    """The camera evidence is insufficient or inconsistent; nothing may move."""


@dataclass(frozen=True)
class SceneParameters:
    rim_height_interval_m: tuple = (.063, .073)   # operator: wall 68 mm +/- 5 mm
    flap_top_m: float = .14                       # operator bound for the folded flaps
    wall_thickness_m: float = .010                # conservative cardboard wall footprint around the opening
    flap_min_extent_m: float = .015               # silhouette beyond the walls counted as a flap
    floor_interval_m: tuple = (0., 0.)            # the box stands on the calibrated plane; cubes are fitted standing on it
    wall_clearance_m: float = .006
    position_sigma_m: float = .002                # camera 2.6 px reprojection + floor height effect
    vertical_sigma_m: float = .004
    side_sigma_m: float = .0015
    yaw_sigma_rad: float = float(np.deg2rad(2.))
    min_cube_score: float = .80
    max_cube_spread_m: float = .003
    max_box_spread_m: float = .012
    max_robot_overlap: float = .05
    workspace_min_x_m: float = .01                # the table's edge nearest the base; what the camera sees beyond it is the floor
    workspace_max_radius_m: float = .45
    unknown_visible_side_m: float = .055          # a visible but unmeasurable cube (tilted against a wall): about a cube's size
    unknown_obstacle_side_m: float = .065         # an object that cannot be measured (clipped, hidden) is kept as a cube this big
    patch_side_m: float = .07
    patch_margin_m: float = .025
    patch_box_clearance_m: float = .06
    patch_reach_m: tuple = (.12, .31)
    flap_planning_margin_m: float = .024
    planning_wall_clearance_m: float = .004
    support_offset_m: float = -.0005              # planner keeps the payload above the support; cubes start ON the table plane
    max_scene_age_s: float = 90.                  # planning only; every phase is re-verified by a fresh scan


@dataclass(frozen=True)
class OverviewScene(ExtractionScene):
    """ExtractionScene whose planning age limit covers the (slow) IK/RRT search."""
    max_age_s: float = 90.

    def require_complete(self, *, now=None, max_age_s=None):
        return super().require_complete(now=now, max_age_s=self.max_age_s if max_age_s is None else max_age_s)


def projection(calibration, z=0.):
    return TableProjection(calibration['camera_matrix'], calibration['distortion'], calibration['camera_from_robot'],
                           z, float(calibration.get('reprojection_rmse_px', 0.)))


def polygons_to_xy(polygons, calibration, z):
    proj = projection(calibration, z)
    return np.array([proj.pixel_to_table(tuple(map(float, p))).xyz_m[:2] for poly in polygons for p in poly])


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def detections(scan, label):
    return [d for d in scan.get('detections', []) if d.get('label') == label]


def robot_mask(backend, joints, calibration, shape, dilate_px=23):
    """Silhouette of the arm meshes in the overview image (as the server computes it)."""
    mask = np.zeros(shape[:2], np.uint8)
    if not joints:
        return mask
    frames = backend.model.frames(joints)
    transform = np.asarray(calibration['camera_from_robot'], float)
    camera = np.asarray(calibration['camera_matrix'], float)
    for link, mesh in backend.meshes.items():
        if link not in frames:
            continue
        world = mesh @ frames[link][:3, :3].T + frames[link][:3, 3]
        optical = world @ transform[:3, :3].T + transform[:3, 3]
        valid = optical[:, 2] > .01
        image = optical[valid] @ camera.T
        uv = np.rint(image[:, :2] / image[:, 2, None]).astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
        mask[uv[ok, 1], uv[ok, 0]] = 255
    return cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8))


def measure_box(scan, calibration, params, views):
    """Box frame, inner opening at the rim plane and conservative outer bound."""
    boxes = [d for d in detections(scan, 'box') if d.get('score', 0.) >= .75]
    openings = [d for d in detections(scan, 'box_opening') if d.get('drop_geometry') is not None and d.get('score', 0.) >= .55]
    if len(boxes) != 1 or len(openings) != 1:
        raise SceneError(f'need exactly one segmented box and one verified opening, saw {len(boxes)} and {len(openings)}')
    opening, box = openings[0], boxes[0]
    low, high = params.rim_height_interval_m
    at_low = polygons_to_xy(opening['silhouette_polygons_px'], calibration, low)
    at_high = polygons_to_xy(opening['silhouette_polygons_px'], calibration, high)
    middle = polygons_to_xy(opening['silhouette_polygons_px'], calibration, .5 * (low + high))
    corners = cv2.boxPoints(cv2.minAreaRect(middle.astype(np.float32)))
    edge = corners[1] - corners[0]
    yaw = float(np.arctan2(edge[1], edge[0]))
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4          # a box is symmetric under 90 degrees
    center = np.asarray(cv2.minAreaRect(middle.astype(np.float32))[0], float)
    rotation = rot_z(yaw)[:2, :2]
    def local(points):
        return (np.asarray(points) - center) @ rotation
    inner_min = np.maximum(local(at_low).min(0), local(at_high).min(0))
    inner_max = np.minimum(local(at_low).max(0), local(at_high).max(0))
    wall_min, wall_max = inner_min - params.wall_thickness_m, inner_max + params.wall_thickness_m
    planes = (0., .5 * (low + high), high, params.flap_top_m)
    outline = {z: local(polygons_to_xy(box['silhouette_polygons_px'], calibration, z)) for z in planes}
    union_min = np.minimum.reduce([o.min(0) for o in outline.values()])
    union_max = np.maximum.reduce([o.max(0) for o in outline.values()])
    flaps, faces = [], []
    for axis in (0, 1):
        # A contour lies either on the table (outer wall face seen by the camera) or on the rim
        # plane (far walls); the smaller of the two overhangs is the one a bare wall explains.
        for sign, label in ((-1, 'min'), (1, 'max')):
            beyond = [(wall_min[axis] - outline[z].min(0)[axis]) if sign < 0 else (outline[z].max(0)[axis] - wall_max[axis]) for z in (0., high)]
            if min(beyond) > params.flap_min_extent_m:
                lo, hi = union_min.copy(), union_max.copy()
                if sign < 0:
                    hi[axis] = wall_min[axis]
                else:
                    lo[axis] = wall_max[axis]
                flaps.append((np.r_[lo, low], np.r_[hi, params.flap_top_m], f'box flap {"xy"[axis]} {label}'))
                faces.append((axis, sign))
    transform = np.eye(4)
    transform[:3, :3] = rot_z(yaw)
    transform[:2, 3] = center
    return FlapAwareBox(
        transform, wall_min, wall_max, inner_min, inner_max, tuple(params.floor_interval_m),
        float(high), float(params.wall_clearance_m),
        {'independent_metric_views': views, 'floor_measured': True, 'opening_verified': True,
         'source': 'overview SAM 3 box silhouette and wall-edge opening, projected to the rim plane interval',
         'floor_assumption': 'calibrated table plane plus cardboard bottom bounded by floor_interval_m',
         'rim_height_interval_m': list(params.rim_height_interval_m), 'opening_score': opening['score'],
         'flaps': [label for _, _, label in flaps], 'flap_top_m': params.flap_top_m,
         'guard_clearance_m': float(params.wall_clearance_m), 'silhouette_min_xy': union_min.tolist(), 'silhouette_max_xy': union_max.tolist()},
        flap_parts=tuple(flaps), flap_faces=tuple(faces))


def spatial_cube(detection, params):
    g = detection['geometry']
    side = float(g['height_m'])
    center = np.asarray(g['center_volume_m'], float)
    axes = np.asarray(g['axes_robot_xy'], float)
    yaw = float(np.arctan2(axes[0, 1], axes[0, 0]))
    vertices = cube_vertices(center[:2], side, yaw, center[2] - side / 2)
    evidence = {
        'geometry_source': 'overview SAM 3 silhouette; all eight vertices fitted on the calibrated table plane',
        'position_sigma_m': [params.position_sigma_m, params.position_sigma_m, params.vertical_sigma_m],
        'side_sigma_m': params.side_sigma_m, 'rotation_sigma_rad': [0., 0., params.yaw_sigma_rad],
        'score': float(detection['score']), 'silhouette_rmse_px': float(detection['geometry_evidence']['silhouette_rmse_px']),
        'image_box_xyxy': list(detection['box_xyxy']), 'robot_overlap_fraction': float(detection.get('robot_overlap_fraction', 0.)),
    }
    return SpatialCube(center, side, rot_z(yaw), vertices, evidence)


def merge_cubes(per_scan, params):
    """Match cubes between scans, require agreement, return one averaged cube each."""
    counts = {len(cubes) for cubes in per_scan}
    if len(counts) != 1:
        raise SceneError(f'cube count changed between observations: {sorted(counts)}')
    merged, spreads = [], []
    for reference in per_scan[0]:
        group = [reference]
        for other in per_scan[1:]:
            group.append(min(other, key=lambda c: np.linalg.norm(c.center_m - reference.center_m)))
        spread = max(np.linalg.norm(a.center_m - b.center_m) for a in group for b in group)
        yaw0 = np.arctan2(group[0].rotation[1, 0], group[0].rotation[0, 0])
        dyaw = [((np.arctan2(c.rotation[1, 0], c.rotation[0, 0]) - yaw0 + np.pi / 4) % (np.pi / 2)) - np.pi / 4 for c in group]
        if spread > params.max_cube_spread_m or max(abs(d) for d in dyaw) > np.deg2rad(3.):
            raise SceneError(f'cube pose is not stable between scans (spread {spread*1000:.1f} mm, yaw {np.rad2deg(max(map(abs, dyaw))):.1f} deg)')
        center = np.mean([c.center_m for c in group], axis=0)
        side = float(np.mean([c.side_m for c in group]))
        yaw = yaw0 + float(np.mean(dyaw))
        vertices = cube_vertices(center[:2], side, yaw, center[2] - side / 2)
        merged.append(SpatialCube(center, side, rot_z(yaw), vertices, group[0].evidence | {'scan_count': len(group), 'spread_m': float(spread)}))
        spreads.append(spread)
    return merged, spreads


def inside_box(cube, box):
    local = np.linalg.inv(box.base_from_box)
    c = local[:3, :3] @ cube.center_m + local[:3, 3]
    return bool(np.all(c[:2] > box.inner_min_xy) and np.all(c[:2] < box.inner_max_xy) and box.floor_interval_m[0] < c[2] < box.obstacle_top_m)


def free_patches(scans, frames, calibration, backend, box, cubes, params, *, limit=8, table_z=0.):
    """Candidate table patches, each verified free and bare in every scan."""
    patches = []
    shape = frames[0].shape
    proj = projection(calibration, table_z)
    base = np.linalg.inv(box.base_from_box)
    masks = []
    for scan, frame in zip(scans, frames):
        occupied = robot_mask(backend, scan.get('robot_joint_deg'), calibration, shape)
        for d in scan.get('detections', []):
            if d.get('label') in {'cube', 'box', 'box_opening', 'background_surface'}:
                grown = np.zeros(shape[:2], np.uint8)
                cv2.fillPoly(grown, [np.asarray(p, np.int32) for p in d.get('silhouette_polygons_px', [])], 255)
                if d.get('label') == 'cube' and not d.get('silhouette_polygons_px'):
                    x1, y1, x2, y2 = map(int, d['box_xyxy'])
                    cv2.rectangle(grown, (x1, y1), (x2, y2), 255, -1)
                occupied |= cv2.dilate(grown, np.ones((15, 15), np.uint8))
        masks.append((occupied, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    half = params.patch_side_m / 2
    candidates = []
    box_center = box.base_from_box[:2, 3]
    for x in np.arange(.10, .34, .02):
        for y in np.arange(-.32, .24, .02):
            radius = float(np.hypot(x, y))
            if not params.patch_reach_m[0] <= radius <= params.patch_reach_m[1]:
                continue
            xy = np.array([x, y])
            local = base[:2, :2] @ xy + base[:2, 3]
            lo = np.asarray(box.evidence['silhouette_min_xy']) - params.patch_box_clearance_m - half
            hi = np.asarray(box.evidence['silhouette_max_xy']) + params.patch_box_clearance_m + half
            if np.all(local > lo) and np.all(local < hi):
                continue
            if any(np.linalg.norm(c.center_m[:2] - xy) < half + params.patch_margin_m + c.side_m for c in cubes):
                continue
            candidates.append((float(np.linalg.norm(xy - box_center)), xy))
    for _, xy in sorted(candidates, key=lambda item: item[0]):
        polygon = np.array([[xy[0] + sx * half, xy[1] + sy * half] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        margin = half + params.patch_margin_m
        outer = np.array([[xy[0] + sx * margin, xy[1] + sy * margin] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        pixels = np.array([proj.table_to_pixel((px, py, table_z)) for px, py in outer])
        if np.any(pixels < 25) or np.any(pixels[:, 0] > shape[1] - 25) or np.any(pixels[:, 1] > shape[0] - 25):
            continue
        region = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(region, np.rint(pixels).astype(np.int32), 255)
        good = True
        for occupied, gray in masks:
            if np.any((region > 0) & (occupied > 0)):
                good = False
                break
            values = gray[region > 0]
            if np.percentile(values, 5) < 110 or values.std() > 32:
                good = False                                  # dark mat, edge, clutter or a shadowed region
                break
        if good:
            patches.append(SupportPatch(polygon, float(table_z + params.support_offset_m), 0., {
                'independent_metric_views': len(scans), 'free_surface_verified': True,
                'method': 'no SAM object or arm silhouette and uniform bright bare table in every overview scan',
                'table_plane_source': calibration.get('table_plane_source'), 'center_xy_m': xy.tolist()}))
            if len(patches) >= limit:
                break
    return patches


def build_scene(scans, frames, calibration, backend, params, *, revision, ids):
    """Merge >= 3 consistent scans into an OverviewScene plus alternative patches."""
    if len(scans) < 3 or len(scans) != len(frames):
        raise SceneError('at least three overview scans with their frames are required')
    boxes = [measure_box(scan, calibration, params, len(scans)) for scan in scans]
    centers = np.array([b.base_from_box[:2, 3] for b in boxes])
    spread = float(max(np.linalg.norm(a - b) for a in centers for b in centers))
    if spread > params.max_box_spread_m:
        raise SceneError(f'box opening is not stable between scans ({spread*1000:.1f} mm)')
    box = boxes[-1]
    box = replace(box, evidence=box.evidence | {'independent_metric_views': len(scans), 'opening_spread_m': spread})
    unknown, cubes, cube_spreads = [], [], []
    # Cluster the cube detections of all scans by image position. A cluster needs a measured
    # metric pose in >= 2 scans (one failed silhouette fit is tolerated); anything less is an
    # unmeasured object and blocks the scene.
    clusters = []
    for scan in scans:
        for d in detections(scan, 'cube'):
            center = np.asarray(d['center_px'], float)
            for cluster in clusters:
                if np.linalg.norm(cluster['center'] - center) <= 15. and cluster['scans'][-1] is not scan:
                    cluster['items'].append((scan, d)); cluster['scans'].append(scan)
                    break
            else:
                clusters.append({'center': center, 'items': [(scan, d)], 'scans': [scan]})
    for cluster in clusters:
        measured = [d for _, d in cluster['items'] if d.get('geometry') is not None and d.get('score', 0.) >= params.min_cube_score
                    and d.get('robot_overlap_fraction', 0.) <= params.max_robot_overlap]
        if len(measured) < 2 or len(cluster['items']) < len(scans) - 1:
            d = cluster['items'][0][1]
            position = d.get('position_robot_m')
            if position is not None and (position[0] < params.workspace_min_x_m or float(np.hypot(position[0], position[1])) > params.workspace_max_radius_m):
                continue          # beyond the table's edge or the arm's reach (the floor behind the base): not on the table, not an obstacle
            if position is not None and len(cluster['items']) >= len(scans) - 1:
                # Something the overview camera sees but cannot measure: clipped by the image border, hidden by the arm, a cube
                # lying tilted against a wall. It stays in the scene as a conservative obstacle (never as a target) instead of
                # blocking everything.
                if True:
                    x1, y1, x2, y2 = d.get('box_xyxy') or (10, 10, 100, 100)
                    clipped = min(x1, y1) <= 2 or x2 >= 1278 or y2 >= 718      # cut by the image border: its place is uncertain
                    side = params.unknown_obstacle_side_m if clipped else params.unknown_visible_side_m
                    sigma = .02 if clipped else .006
                    center = np.array([position[0], position[1], side / 2.])
                    evidence = {'geometry_source': 'unmeasured object kept as a conservative obstacle', 'unmeasured': True,
                                'position_sigma_m': [sigma] * 3, 'side_sigma_m': .008, 'rotation_sigma_rad': [0., 0., .3 if clipped else .1],
                                'center_px': [float(v) for v in d['center_px']], 'reason': d.get('unconfirmed_reason') or 'not measured consistently'}
                    cubes.append(SpatialCube(center, side, np.eye(3), cube_vertices(center[:2], side, 0., 0.), evidence))
                    cube_spreads.append(0.)
                    continue
            unknown.append({'box_xyxy': d.get('box_xyxy'), 'seen_in_scans': len(cluster['items']), 'measured_in_scans': len(measured),
                            'reason': d.get('unconfirmed_reason') or d.get('projection_error') or 'not measured consistently'})
            continue
        merged, spreads = merge_cubes([[spatial_cube(d, params)] for d in measured], params)
        cubes.append(merged[0])
        cube_spreads.append(spreads[0])
    registry = ids.assign(cubes)
    patches = free_patches(scans, frames, calibration, backend, box, cubes, params)
    last = scans[-1]['frame']['captured_at_unix_s']
    placeholder = patches[0] if patches else SupportPatch(
        np.array([[0., 0.], [.07, 0.], [.07, .07], [0., .07]]), 0., .002, {'independent_metric_views': 3, 'free_surface_verified': True})
    scene = OverviewScene(revision, float(last), box, registry, placeholder, tuple(unknown), not unknown and bool(patches),
                          max_age_s=params.max_scene_age_s)
    return scene, patches, {'box_spread_m': spread, 'cube_spreads_m': cube_spreads, 'unknown': unknown,
                            'patch_candidates': len(patches), 'inside_box': [k for k, c in registry.items() if inside_box(c, box)]}


class IdentityTracker:
    """Stable cube-N identifiers by position (8 mm) and size (3 mm)."""
    def __init__(self):
        self.known, self.counter = {}, 0

    def assign(self, cubes):
        result, free = {}, set(self.known)
        for cube in cubes:
            matches = [k for k in free if np.linalg.norm(self.known[k].center_m - cube.center_m) < .008 and abs(self.known[k].side_m - cube.side_m) < .003]
            if len(matches) > 1:
                raise SceneError('cube identity is ambiguous')
            if matches:
                key = matches[0]
                free.discard(key)
            else:
                key = f'cube-{self.counter}'
                self.counter += 1
            result[key] = cube
        self.known = dict(result)
        return result
