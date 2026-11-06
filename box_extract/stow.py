"""Stowing: a cube from the table is carried into the box and set down on its floor.

The same phases as the extraction, run the other way: approach, enter (descend onto the table cube), grasp, lift above the
rim, transfer over the rim, lower onto a free patch of the box floor, release, retract. The planner, the collision world and
the driver are the extraction's; only the target (a cube outside the box) and the support (a patch of the box floor) differ.
"""
from __future__ import annotations

import numpy as np
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld, SupportPatch

from .planning import TunedExtractionPlanner
from .scene_builder import inside_box

FLOOR_PATCH_SIDE_M = .09          # a cube turned by any yaw fits (diagonal 62 mm) with the support's own 5 mm margin
# Keep-outs differ along the two sides of the box because the gripper is not round: its fingers span 86 mm along the closing
# axis and are 52 mm across. The closing axis is put across the short side, so along the long side only the 52 mm matters.
WALL_KEEPOUT_LONG_M, WALL_KEEPOUT_SHORT_M = .034, .050      # patch centre to wall
CUBE_KEEPOUT_LONG_M, CUBE_KEEPOUT_SHORT_M = .057, .085      # patch centre to a cube already inside


def box_floor_patches(scene, params, *, limit=8):
    """Candidate patches on the floor of the box, away from its walls and from the cubes already inside."""
    box = scene.box
    inner_lo, inner_hi = np.asarray(box.inner_min_xy, float), np.asarray(box.inner_max_xy, float)
    long_axis = int(np.argmax(inner_hi - inner_lo))
    wall = np.zeros(2); keep = np.zeros(2)
    wall[long_axis], wall[1 - long_axis] = WALL_KEEPOUT_LONG_M, WALL_KEEPOUT_SHORT_M
    keep[long_axis], keep[1 - long_axis] = CUBE_KEEPOUT_LONG_M, CUBE_KEEPOUT_SHORT_M
    lo, hi = inner_lo + wall, inner_hi - wall
    centre_box = (inner_lo + inner_hi) / 2.
    if np.any(hi < lo):
        lo = hi = centre_box
    inverse = np.linalg.inv(box.base_from_box)
    others = [(inverse[:3, :3] @ c.center_m + inverse[:3, 3])[:2] for c in scene.cubes.values() if inside_box(c, box)]
    floor_z = float((box.base_from_box @ np.r_[0., 0., box.floor_interval_m[1], 1.])[2]) + params.support_offset_m
    # Slots along the long side at the cube spacing, centre first, then away from the robot's base, then toward it: three
    # cubes fit in a box this long, and the one nearest the base wall (hardest to take out again) goes in last.
    spacing = CUBE_KEEPOUT_LONG_M + .0005
    sign = 1. if box.base_from_box[:2, :2][0, long_axis] >= 0. else -1.      # +k moves toward larger robot x
    mid = (lo + hi) / 2.
    slots = []
    for k in range(-4, 5):
        p = mid.copy()
        p[long_axis] = mid[long_axis] + sign * k * spacing
        if np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9):
            for dy in (0., -.005, .005):
                q = p.copy()
                q[1 - long_axis] = np.clip(mid[1 - long_axis] + dy, lo[1 - long_axis], hi[1 - long_axis])
                slots.append((abs(k), 0 if k > 0 else 1, abs(dy), q))
    candidates = [(i, q) for i, (_, _, _, q) in enumerate(sorted(slots, key=lambda t: t[:3])) if not any(np.all(np.abs(q - o) < keep) for o in others)]
    patches, half = [], FLOOR_PATCH_SIDE_M / 2.
    for _, p in sorted(candidates, key=lambda item: item[0]):
        corners = np.array([[p[0] + sx * half, p[1] + sy * half] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        base_xy = (box.base_from_box[:2, :2] @ corners.T).T + box.base_from_box[:2, 3]
        centre_xy = box.base_from_box[:2, :2] @ p + box.base_from_box[:2, 3]
        if any(np.linalg.norm(centre_xy - q.evidence['center_xy_m']) < .02 for q in patches):
            continue
        patches.append(SupportPatch(base_xy, floor_z, 0., {'independent_metric_views': 3, 'free_surface_verified': True, 'center_xy_m': centre_xy.tolist(),
                                                           'method': 'floor of the measured box, away from its walls and from the cubes inside it'}))
        if len(patches) >= limit:
            break
    return patches


class StowPlanner(TunedExtractionPlanner):
    """Plans to move one cube from the table into the box (the parent plans the reverse)."""
    stowing = True

    def plan(self, scene, current):
        self.check_cancel()
        scene.require_complete()
        if scene.box.base_from_box[2, 2] < .999:
            raise ValueError('this stowing requires a measured level box; tilted boxes are unsupported')
        world = ExtractionCollisionWorld(self.backend, scene)
        centre = scene.box.base_from_box[:2, 3]
        targets = [(k, c) for k, c in scene.cubes.items() if not inside_box(c, scene.box) and not c.evidence.get('unmeasured')]
        rejections = []
        for target_id, cube in sorted(targets, key=lambda item: np.linalg.norm(item[1].center_m[:2] - centre)):
            try:
                for grasp, closed in self.grasp_candidates(scene, cube, current):
                    try:
                        return self._candidate(scene, current, target_id, cube, grasp, closed, world)
                    except ValueError as exc:
                        rejections.append(str(exc))
            except ValueError as exc:
                rejections.append(str(exc))
        raise ValueError('no collision-free measured grasp and placement: ' + ('; '.join(rejections[-3:]) or 'no cube outside the box'))
