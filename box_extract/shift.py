"""Follow a box that was nudged: shift the scene and the pending pre-grasp path by the measured displacement.

The overview camera measures how far the box's left rim moved. The cubes inside move with the box. The path
that has not been driven yet (the rest of the approach, the entry and the jaw closure) is the same path
translated by that displacement with the gripper orientation kept (`planner.translated`), the approach
ramping from zero at the arm's current pose so it stays continuous. Every translated waypoint is checked
against the shifted world, also under the gravity-compensated command. Phases after the grasp are planned
from the scene as it is then and need no correction. No new search is run.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .scene_builder import inside_box

MIN_SHIFT_M = .0012
MAX_SHIFT_M = .025


def shift_scene(scene, d_xy):
    """The scene with the box and the cubes lying in it moved by `d_xy` (metres, robot frame)."""
    d = np.r_[np.asarray(d_xy, float), 0.]
    transform = scene.box.base_from_box.copy()
    transform[:3, 3] += d
    cubes = {}
    for key, cube in scene.cubes.items():
        moved = inside_box(cube, scene.box)
        cubes[key] = replace(cube, center_m=cube.center_m + d, vertices_m=cube.vertices_m + d) if moved else cube
    return replace(scene, box=replace(scene.box, base_from_box=transform), cubes=cubes)


def correct_pending(planner, world, pending, d_xy, target_id):
    """Translate the pre-grasp phases in `pending`; raise ValueError if the translated path is not collision-free."""
    d = np.r_[np.asarray(d_xy, float), 0.]
    out = []
    for phase in pending:
        if phase.name == 'approach':
            n = len(phase.joints)
            joints = [q if i == 0 else planner.translated(q, d * (i / (n - 1))) for i, q in enumerate(phase.joints)]
            policy = {}
        elif phase.name in ('enter', 'grasp'):
            joints = [planner.translated(q, d) for q in phase.joints]
            policy = {'target': target_id, 'jaw_contact': True}
        else:
            out.append(phase)
            continue
        for q in joints:
            for pose in (q, planner._command_pose(q)):
                reason = world.reason(pose, **policy)
                if reason and not (phase.name == 'grasp' and reason.endswith(f'cube {target_id}')):
                    raise ValueError(f'{phase.name}: {reason}')
        out.append(replace(phase, joints=type(phase.joints)(joints)))
    return out
