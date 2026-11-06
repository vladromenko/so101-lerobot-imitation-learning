"""Overview-camera perception: SAM 3 scans -> scene, scene-change interlock, placement check."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from .opening_fallback import estimate_opening
from .scene_builder import (IdentityTracker, SceneError, build_scene, detections, inside_box, projection, robot_mask)


MIN_RIM_SCORE = .6        # weaker matches (shadows, the arm in view) are not trusted and never stop the run
MIN_SHIFT_PX = 2.         # about 1.2 mm at the rim plane: below the matching noise
MAX_SHIFT_PX = 40.        # about 24 mm: a box moved further than the path can follow is a new scene


class SceneChanged(RuntimeError):
    """Something moved or appeared; the arm must hold its pose."""


def api_json(api, path, method='GET', timeout=120.):
    request = Request(api + path, method=method, data=b'' if method == 'POST' else None)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f'{method} {path}: HTTP {exc.code}: {exc.read().decode(errors="replace")[:300]}') from exc


class OverviewPerception:
    def __init__(self, api, calibration, backend, params, *, scans=3, log=print):
        self.api, self.calibration, self.backend, self.params = api, calibration, backend, params
        self.scans, self.log = scans, log
        self.ids = IdentityTracker()
        self.reference = None            # image-space reference of the planned scene
        self.revision = 0

    # -- raw measurements ------------------------------------------------------------------
    def scan(self):
        record = api_json(self.api, '/api/scan-hybrid', 'POST')
        frame = cv2.imread(str(Path(record['artifacts']) / 'frame.jpg'))
        if frame is None:
            raise RuntimeError('the scan frame is not readable')
        if not any(d['label'] == 'box_opening' for d in record['detections']):
            opening, evidence = estimate_opening(frame, record)
            if opening is not None:
                record['detections'].append(opening)
                self.log(f'no opening from the server; rim-edge fallback: {evidence["corners_px"]}' + (' (held from before: the rim is hidden)' if evidence.get('remembered') else f' support {[round(v, 2) for v in evidence["support"]]}'))
            else:
                self.log(f'no opening from the server and none from the rim edges: {evidence}')
        return record, frame

    def survey(self, attempts=3, mode='extract'):
        """Three agreeing scans -> (scene, placement patches, diagnostics)."""
        failures = []
        for attempt in range(attempts):
            pairs = [self.scan() for _ in range(self.scans)]
            scans, frames = [p[0] for p in pairs], [p[1] for p in pairs]
            try:
                self.revision += 1
                scene, patches, info = build_scene(scans, frames, self.calibration, self.backend, self.params,
                                                   revision=f'overview-{self.revision}', ids=self.ids)
            except SceneError as exc:
                failures.append(str(exc))
                self.log(f'survey attempt {attempt + 1}: {exc}')
                continue
            self.reference = self._reference(scans[-1], scene, frames[-1])
            info['scans'] = [s['artifacts'] for s in scans]
            if mode == 'stow':
                from .stow import box_floor_patches
                patches = box_floor_patches(scene, self.params)
                scene = replace(scene, coverage_verified=not scene.unknown_candidates and bool(patches))   # coverage of the box floor, not of the table
            return scene, patches, info
        raise SceneError('no consistent scene after %d attempts: %s' % (attempts, '; '.join(failures[-2:])))

    # -- helpers --------------------------------------------------------------------------
    @staticmethod
    def box_anchor(scan):
        """Where the box is in the image, from its most stable features.

        The SAM silhouette centre (lid included) wanders by several pixels with the arm's shadow, and the right
        edge of the measured opening wanders with it; the left and top rim lines stay within about 3 px. The anchor
        is the mean of the opening's top-left and bottom-left corners; without an opening, the silhouette centre.
        """
        openings = [d for d in detections(scan, 'box_opening') if d.get('silhouette_polygons_px')]
        if len(openings) == 1 and len(openings[0]['silhouette_polygons_px'][0]) == 4:
            quad = np.asarray(openings[0]['silhouette_polygons_px'][0], float)
            return (quad[0] + quad[3]) / 2.
        return np.asarray(detections(scan, 'box')[0]['center_px'], float)

    @staticmethod
    def rim_gradient(frame):
        """Edge strength of the frame: insensitive to the arm's shadow, which changes brightness but not edges."""
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    @staticmethod
    def rim_template(scan, frame):
        """Grey patch of the box's left rim and inner wall (the side the arm never covers) and where it was taken."""
        openings = [d for d in detections(scan, 'box_opening') if d.get('silhouette_polygons_px')]
        if len(openings) != 1 or len(openings[0]['silhouette_polygons_px'][0]) != 4:
            return None
        quad = np.asarray(openings[0]['silhouette_polygons_px'][0], float)
        x0, x1 = int(min(quad[0][0], quad[3][0])) - 30, int(max(quad[0][0], quad[3][0])) + 30
        y0, y1 = int(quad[0][1]) + 15, int(quad[3][1]) - 15
        return {'patch': OverviewPerception.rim_gradient(frame)[y0:y1, x0:x1].copy(), 'origin': (x0, y0)}

    def rim_shift(self, frame, occluder):
        """How far the box's left rim moved in the image since planning, ignoring pixels the arm covers.

        Returns None without a template, {'verifiable': False} when the arm hides most of the rim, otherwise
        {'verifiable': True, 'shift_px': (dx, dy), 'score': normalised correlation over the visible pixels}.
        """
        template = self.reference.get('box_template')
        if template is None:
            return None
        patch, (x0, y0) = template['patch'], template['origin']
        height, width = patch.shape
        visible = occluder[y0:y0 + height, x0:x0 + width] == 0
        if visible.shape != patch.shape or visible.mean() < .3:
            return {'verifiable': False, 'visible': float(visible.mean()) if visible.size else 0.}
        pad = 40
        gradient = self.rim_gradient(frame)
        wx0, wy0 = max(0, x0 - pad), max(0, y0 - pad)
        window = gradient[wy0:y0 + height + pad, wx0:x0 + width + pad]
        result = cv2.matchTemplate(window, patch, cv2.TM_CCORR_NORMED, mask=visible.astype(np.float32))
        _, _, _, location = cv2.minMaxLoc(np.nan_to_num(result, nan=0., posinf=0., neginf=0.))
        dx, dy = location[0] + wx0 - x0, location[1] + wy0 - y0
        found = window[location[1]:location[1] + height, location[0]:location[0] + width]
        a, b = patch[visible], found[visible]
        denominator = float(a.std() * b.std())
        score = float(np.mean((a - a.mean()) * (b - b.mean())) / denominator) if denominator > 1e-6 else 0.
        return {'verifiable': True, 'shift_px': (float(dx), float(dy)), 'score': score, 'visible': float(visible.mean())}

    def accept_box_shift(self, shift_px):
        """The box now sits where it was found: later checks compare against that."""
        dx, dy = int(round(shift_px[0])), int(round(shift_px[1]))
        template = self.reference['box_template']
        template['origin'] = (template['origin'][0] + dx, template['origin'][1] + dy)
        self.reference['box_px'] = self.reference['box_px'] + np.array([dx, dy], float)
        for key in self.reference['inside']:
            self.reference['cube_px'][key] = self.reference['cube_px'][key] + np.array([dx, dy], float)

    def px_to_xy(self, shift_px):
        """Robot-frame displacement (m) at the rim plane for an image shift at the template's centre."""
        template = self.reference['box_template']
        centre = np.array(template['origin'], float) + np.array(template['patch'].shape[::-1], float) / 2.
        proj = projection(self.calibration, float(np.mean(self.params.rim_height_interval_m)))
        a = np.asarray(proj.pixel_to_table(tuple(centre)).xyz_m, float)[:2]
        b = np.asarray(proj.pixel_to_table(tuple(centre + np.asarray(shift_px, float))).xyz_m, float)[:2]
        return np.asarray(b - a, float)

    def _reference(self, scan, scene, frame):
        boxes = detections(scan, 'box')
        cubes = [d for d in detections(scan, 'cube') if d.get('geometry')]
        cube_px = {}
        for key, cube in scene.cubes.items():
            if cube.evidence.get('unmeasured'):
                cube_px[key] = np.asarray(cube.evidence['center_px'], float)
                continue
            best = min(cubes, key=lambda d: np.linalg.norm(np.asarray(d['geometry']['center_volume_m'][:2]) - cube.center_m[:2]))
            cube_px[key] = np.asarray(best['center_px'], float)
        return {'box_px': self.box_anchor(scan), 'cube_px': cube_px, 'box_template': self.rim_template(scan, frame),
                'inside': set(self.inside_ids(scene)), 'unmeasured': {k for k, c in scene.cubes.items() if c.evidence.get('unmeasured')}}

    def inside_ids(self, scene):
        """Measured cubes in the box (an unmeasurable object is an obstacle, not a cube to count or to take)."""
        return [k for k, c in scene.cubes.items() if inside_box(c, scene.box) and not c.evidence.get('unmeasured')]

    @staticmethod
    def measured(scene):
        return {k: c for k, c in scene.cubes.items() if not c.evidence.get('unmeasured')}

    # -- interlocks -------------------------------------------------------------------------
    def assert_unchanged(self, scene, *, target_id, joints, carrying, arm_over_box):
        """Compare a fresh scan with the planned scene, ignoring what the arm itself hides."""
        scan, frame = self.scan()
        mask = robot_mask(self.backend, joints, self.calibration, frame.shape)
        near_arm = cv2.dilate(mask, np.ones((61, 61), np.uint8))
        boxes = detections(scan, 'box')
        if len(boxes) != 1:
            raise SceneChanged('the box is not seen as a single object')
        rim = self.rim_shift(frame, near_arm)
        shift_px = np.zeros(2)
        box_state = 'rim hidden by the arm: box position not re-measured'
        if rim is None:
            tolerance = 14. if arm_over_box else 6.
            shift = float(np.linalg.norm(self.box_anchor(scan) - self.reference['box_px']))
            if shift > tolerance:
                raise SceneChanged(f'the box moved ({shift:.1f} px)')
            box_state = 'anchor check'
        elif rim['verifiable'] and rim['score'] >= MIN_RIM_SCORE:
            shift_px = np.asarray(rim['shift_px'], float)
            if np.linalg.norm(shift_px) > MAX_SHIFT_PX:
                raise SceneChanged(f'the box moved too far ({np.linalg.norm(shift_px):.0f} px)')
            box_state = f'rim matched (correlation {rim["score"]:.2f})'
        elif rim['verifiable']:
            box_state = f'rim match too weak to trust (correlation {rim["score"]:.2f}, arm in view): box position not re-measured'
        shift = float(np.linalg.norm(shift_px))
        found = [d for d in detections(scan, 'cube') if d.get('score', 0.) >= .6]
        matched = set()
        for key, expected in self.reference['cube_px'].items():
            expected = expected + (shift_px if key in self.reference['inside'] else 0.)
            u, v = int(expected[0]), int(expected[1])
            hidden = mask[v, u] > 0                           # the arm silhouette covers this cube
            loose = key in self.reference['unmeasured']        # an obstacle without a measured pose: found within a cube's size, or let go
            hit = [i for i, d in enumerate(found) if np.linalg.norm(np.asarray(d['center_px'], float) - expected) <= (30. if loose else 12.)]
            if hit:
                matched.update(hit)
                continue
            if loose:
                continue
            if key == target_id and (carrying or arm_over_box):
                continue                                  # the cube being handled moves with / hides under the gripper
            if not hidden:
                raise SceneChanged(f'{key} is no longer where it was measured')
        for i, d in enumerate(found):
            if i in matched:
                continue
            u, v = map(int, d['center_px'])
            if any(np.linalg.norm(np.asarray(d['center_px'], float) - self.reference['cube_px'][k]) <= 45. for k in self.reference['unmeasured']):
                continue
            if 0 <= v < near_arm.shape[0] and 0 <= u < near_arm.shape[1] and near_arm[v, u] > 0 and (carrying or arm_over_box):
                continue
            raise SceneChanged(f'a new cube-like object appeared at pixel ({u}, {v})')
        correction = None
        if shift >= MIN_SHIFT_PX:
            correction = self.px_to_xy(shift_px)
        return {'box_shift_px': shift, 'cubes_seen': len(found), 'box': box_state, 'box_shift_xy_m': correction, 'box_shift_px_vec': shift_px}

    def verify_stowed(self, plan, inside_before, total_before):
        """After parking: the target is in the box, at the planned place on its floor, and nothing else changed."""
        scene, _, info = self.survey()
        inside = self.inside_ids(scene)
        if len(inside) != inside_before + 1:
            raise RuntimeError(f'cubes in the box: {len(inside)}, expected {inside_before + 1}')
        if len(self.measured(scene)) != total_before:
            raise RuntimeError(f'cube count changed: {len(self.measured(scene))} instead of {total_before}')
        placed = [c for k, c in scene.cubes.items() if k in inside and np.linalg.norm(c.center_m[:2] - plan.placement_center_m[:2]) <= .03]
        if len(placed) != 1:
            raise RuntimeError('the carried cube was not found at the planned place on the box floor')
        return {'placement_error_m': float(np.linalg.norm(placed[0].center_m[:2] - plan.placement_center_m[:2])), 'scene': info}

    def verify_placement(self, plan, inside_before, total_before):
        """After parking: the target is on the table at the planned spot and out of the box."""
        scene, _, info = self.survey()
        inside = self.inside_ids(scene)
        if len(inside) != inside_before - 1:
            raise RuntimeError(f'cubes in the box: {len(inside)}, expected {inside_before - 1}')
        if len(self.measured(scene)) != total_before:
            raise RuntimeError(f'cube count changed: {len(self.measured(scene))} instead of {total_before}')
        placed = [c for k, c in scene.cubes.items() if k not in inside and np.linalg.norm(c.center_m[:2] - plan.placement_center_m[:2]) <= .012]
        if len(placed) != 1:
            raise RuntimeError('the extracted cube was not found at the planned place on the table')
        return {'placement_error_m': float(np.linalg.norm(placed[0].center_m[:2] - plan.placement_center_m[:2])), 'scene': info}
