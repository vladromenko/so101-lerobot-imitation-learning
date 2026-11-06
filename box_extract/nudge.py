"""Moving a cube inside the box with the closed gripper when no grasp of it can be planned where it lies.

The box is narrow and often stands at the edge of the arm's reach, so only a small part of its floor admits a grasp
(the jaws, the forearm and the near flap all have to fit). When no grasp entry exists for a cube, the arm pushes it along a box
axis to a place where one does: the closed jaws approach from above the rim, go down beside the cube, move it flat along the
floor, back off and leave upwards. Everything is planned and checked like the extraction itself (collision meshes, gravity
compensated command, tracking lag, simulated replay); afterwards the scene is measured again and the extraction is planned as usual.
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from so101_pick_place.experiments.box_extraction.planner import ExtractionPlan, Phase
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld

from .planning import TunedExtractionPlanner

PUSH_JAW = 5.             # closed jaws, percent of the gripper's range
START_GAP_M = .008        # the jaws stop this far from the cube before the push
BACKOFF_M = .008          # after the push the jaws step back this far before rising
STEP_M = .002
BOTTOM_ABOVE_FLOOR_M = .018     # the model keeps its table clearance (gap) under the jaws


@dataclass(frozen=True)
class PushPhase(Phase):
    drop_target: bool = False       # the cube moves during this phase: it is left out of the collision world


def moved_cube(scene, cube, xy, yaw_in_box=0.):
    """The cube standing at `xy` on the floor, its faces turned `yaw_in_box` against the box walls."""
    box_yaw = float(np.arctan2(scene.box.base_from_box[1, 0], scene.box.base_from_box[0, 0])) + yaw_in_box
    rotation = np.array([[np.cos(box_yaw), -np.sin(box_yaw), 0.], [np.sin(box_yaw), np.cos(box_yaw), 0.], [0., 0., 1.]])
    half, center = cube.side_m / 2., np.r_[np.asarray(xy, float), cube.center_m[2]]
    unit = np.array([[x, y, z] for z in (-half, half) for x, y in ((-half, -half), (half, -half), (half, half), (-half, half))])
    return replace(cube, center_m=center, rotation=rotation, vertices_m=unit @ rotation.T + center)


class NudgePlanner(TunedExtractionPlanner):
    pusher_axis_tolerance_deg = 8.   # how far the jaws' flat side may turn from facing the push
    nudges = 0                       # pushes made since the last cube was taken out
    pusher_pitches_deg = (-90., -70., -50.)
    min_pusher_down = .5
    # ------------------------------------------------------------------ is there a grasp at all?
    def entry_report(self, view, target_id, current, *, limit=12):
        """(entries that survive, entries tried, reasons): the vertical entry of every grasp candidate against the world."""
        cube = view.cubes[target_id]
        world = ExtractionCollisionWorld(self.backend, view)
        policy = {'target': target_id, 'jaw_contact': True}
        rim = (view.box.base_from_box @ np.r_[0., 0., view.box.obstacle_top_m, 1.])[2]
        rise = max(.025, rim + view.box.clearance_m + self.rim_margin_m + cube.side_m / 2 - cube.center_m[2])
        tried, ok, why = 0, 0, collections.Counter()
        try:
            for grasp, _closed in self.grasp_candidates(view, cube, current):
                tried += 1
                frame = self.model.frames(grasp)['gripper_frame_link']
                local = (cube.vertices_m - frame[:3, 3]) @ frame[:3, :3]
                try:
                    up = self.vertical_center(grasp, rise, local.mean(0), world, **policy)
                    bad = next((f'command {r}' for q in up if (r := world.reason(self._command_pose(q), **policy))), None) \
                        or self._tracking_robust(up, world, **policy)
                except ValueError as exc:
                    bad = str(exc)
                if bad:
                    why[bad.split(': ', 1)[-1][:60]] += 1
                else:
                    ok += 1
                    break
                if tried >= limit:
                    break
        except ValueError as exc:
            why['no candidates: ' + str(exc)[:60]] += 1
        return ok, tried, why

    def graspable(self, view, ids, current):
        return [k for k in ids if self.entry_report(view, k, current, limit=40)[0]]

    # ------------------------------------------------------------------ the pusher
    def _jaw_points_local(self, joints):
        frames = self.model.frames(joints)
        tool = frames['gripper_frame_link']
        parts = [(self.backend._transform_points(frames[n], self.backend.meshes[n]) - tool[:3, 3]) @ tool[:3, :3]
                 for n in ('gripper_link', 'moving_jaw_so101_v1_link')]
        return np.vstack(parts)

    def _surface_offsets(self, joints, direction, lateral, side_m):
        """Closed-jaw surface relative to the tool origin in the base axes: (lowest z, leading extent along `direction`, lateral centre)."""
        tool = self.model.frames(joints)['gripper_frame_link']
        offsets = self._jaw_points_local(joints) @ tool[:3, :3].T
        low = float(offsets[:, 2].min())
        band = (offsets[:, 2] >= low + .001) & (offsets[:, 2] <= low + side_m - BOTTOM_ABOVE_FLOOR_M - .004)
        if not band.any():
            return None
        lead = float((offsets[band] @ direction).max())
        front = band & (offsets @ direction >= lead - .01)
        lat = offsets[front] @ lateral
        return low, lead, float((lat.min() + lat.max()) / 2.)

    def pusher_poses(self, cube, direction, floor_z):
        """Closed-jaw poses `START_GAP_M` behind the cube's trailing side, centred on it, the jaw bottom just above the floor.

        Like the grasp search, the tilt of the tool is free: every seed is refined by least squares on the tool position and on the
        direction of its closing axis (its flat jaw side faces the push)."""
        direction = np.asarray(direction, float)
        lateral = np.array([-direction[1], direction[0], 0.])
        trailing = float((cube.vertices_m @ direction).min())
        centre = cube.center_m
        self.pusher_rejections = collections.Counter()
        reference = None
        for sign in (1., -1.):
            axis = sign * direction
            for pitch in self.pusher_pitches_deg:
                for roll in np.linspace(*self.limits[4], 9):
                    p = np.r_[centre[:2] - direction[:2] * (cube.side_m / 2. + START_GAP_M + .02), floor_z + BOTTOM_ABOVE_FLOOR_M + .012]
                    seed = None
                    for shrink in (0., .01, .02, .03, .04):          # a seed nearer the base when the exact point has no closed-form pose
                        try:
                            q = p - np.r_[p[:2] / np.linalg.norm(p[:2]) * shrink, 0.]
                            seed = self.model.ik(*q, pitch=pitch, roll=float(roll), gripper=PUSH_JAW)
                            break
                        except self.backend._unreachable:
                            continue
                    if seed is None:
                        self.pusher_rejections['no seed'] += 1
                        continue
                    x = self.vector(seed.deg)
                    joints = None
                    for iteration in range(3):
                        def residual(v, target=p):
                            frame = self.model.frames(self.joints(v, PUSH_JAW))['gripper_frame_link']
                            return np.r_[(frame[:3, 3] - target) * 1000., (frame[:3, 0] - axis) * (30. if self.pusher_axis_tolerance_deg <= 8. else 6.), max(0., self.min_pusher_down + frame[2, 2]) * 30.]
                        fit = least_squares(residual, x, bounds=(self.limits[:, 0], self.limits[:, 1]), max_nfev=80)
                        error = residual(fit.x)
                        if not np.all(np.isfinite(fit.x)) or np.linalg.norm(error[:3]) > 1.5 or np.linalg.norm(error[3:6]) / (30. if self.pusher_axis_tolerance_deg <= 8. else 6.) > np.sin(np.deg2rad(self.pusher_axis_tolerance_deg)):
                            self.pusher_rejections['position/axis not reached'] += 1
                            break
                        x = fit.x
                        joints = self.joints(x, PUSH_JAW)
                        surface = self._surface_offsets(joints, direction, lateral, cube.side_m)
                        if surface is None:
                            self.pusher_rejections['no jaw surface in the cube band'] += 1
                            joints = None
                            break
                        low, lead, lat = surface
                        new = (trailing - START_GAP_M - lead) * direction + (float(centre @ lateral) - lat) * lateral + np.array([0., 0., floor_z + BOTTOM_ABOVE_FLOOR_M - low])
                        if np.linalg.norm(new - p) < 5e-4:
                            break
                        p = new
                    else:
                        self.pusher_rejections['position did not settle'] += 1
                        joints = None
                    if joints is None:
                        continue
                    frame = self.model.frames(joints)['gripper_frame_link']
                    if frame[2, 2] > -self.min_pusher_down + 1e-3:
                        self.pusher_rejections['tool not pointing down'] += 1
                        continue
                    yield joints

    MAX_JOINT_STEP_DEG = 2.5          # the arm's follower refuses neighbouring poses further apart than 3 degrees

    def _line(self, start, delta, world=None, *, hold_fingers=False, **policy):
        """The tool origin on a straight line of `delta` metres; the jaws' flat side keeps facing the same way, the tilt of the tool
        may drift only a few degrees (a five-joint arm cannot hold a full orientation along every line). Steps are 2 mm, shortened wherever the joints
        would otherwise jump."""
        delta = np.asarray(delta, float)
        length = float(np.linalg.norm(delta))
        frame = self.model.frames(start)['gripper_frame_link']
        axis, fingers, origin = frame[:3, 0].copy(), frame[:3, 2].copy(), frame[:3, 3].copy()
        path, previous = [start], start
        fraction, step, base = 0., min(1., STEP_M / max(length, 1e-9)), min(1., STEP_M / max(length, 1e-9))
        while fraction < 1. - 1e-9:
            self.check_cancel()
            reach = min(1., fraction + step)
            target = origin + delta * reach
            seed = self.vector(previous)

            def residual(q):
                actual = self.model.frames(self.joints(q, start['gripper']))['gripper_frame_link']
                return np.r_[(actual[:3, 3] - target) * 1000., (actual[:3, 0] - axis) * 20., (actual[:3, 2] - fingers) * (10. if hold_fingers else 0.), .1 * (q - seed)]
            fit = least_squares(residual, seed, bounds=(self.limits[:, 0], self.limits[:, 1]), max_nfev=100)
            final = residual(fit.x)
            good = (np.all(np.isfinite(fit.x)) and np.linalg.norm(final[:3]) <= .75 and np.linalg.norm(final[3:6]) / 20. <= np.sin(np.deg2rad(8.)) and (not hold_fingers or np.linalg.norm(final[6:9]) / 10. <= np.sin(np.deg2rad(12.)))
                    and float(np.abs(fit.x - seed).max()) <= self.MAX_JOINT_STEP_DEG)
            if not good:
                step /= 2.
                if step * length < 2e-4:
                    raise ValueError(f'no continuous IK keeps the jaws on the planned line (at {fraction * length * 1000:.0f} of {length * 1000:.0f} mm: position {np.linalg.norm(final[:3]):.2f} mm, jaw side {np.rad2deg(np.arcsin(min(1., np.linalg.norm(final[3:6]) / 20.))):.1f} deg, finger axis {np.rad2deg(np.arcsin(min(1., np.linalg.norm(final[6:9]) / 10.))):.1f} deg, joints {float(np.abs(fit.x - seed).max()):.1f} deg)')
                continue
            goal = self.joints(fit.x, start['gripper'])
            path.extend(self.edge(previous, goal, world, **policy)[1:] if world is not None else [goal])
            previous, fraction, step = goal, reach, min(base, step * 1.5)
        return tuple(path)

    # ------------------------------------------------------------------ the whole push
    PLAN_SECONDS = 90.

    def plan_push(self, view, current, target_id, direction, distance):
        """Plan one push, within `PLAN_SECONDS`."""
        deadline, outer = time.time() + self.PLAN_SECONDS, self.check_cancel

        def check():
            outer()
            if time.time() > deadline:
                raise ValueError(f'no push plan within {self.PLAN_SECONDS:.0f} s')
        self.check_cancel = check
        try:
            return self._plan_push(view, current, target_id, direction, distance)
        finally:
            self.check_cancel = outer

    def _plan_push(self, view, current, target_id, direction, distance):
        cube = view.cubes[target_id]
        direction = np.asarray(direction, float)
        box = view.box
        floor_z = float((box.base_from_box @ np.r_[0., 0., box.floor_interval_m[1], 1.])[2])
        rim = float((box.base_from_box @ np.r_[0., 0., box.obstacle_top_m, 1.])[2])
        world = ExtractionCollisionWorld(self.backend, view)
        bare = ExtractionCollisionWorld(self.backend, replace(view, cubes={k: v for k, v in view.cubes.items() if k != target_id}))
        policy = {'target': target_id, 'jaw_contact': True}
        closed = dict(current, gripper=PUSH_JAW)
        rejections = []
        for start in self.pusher_poses(cube, direction, floor_z):
            try:
                tip_z = self.model.frames(start)['gripper_frame_link'][2, 3]
                rise = max(.03, rim + box.clearance_m + self.rim_margin_m + .006 - (floor_z + BOTTOM_ABOVE_FLOOR_M))
                up = self._line(start, [0., 0., rise], world, **policy)
                enter = up[::-1]
                push = self._line(start, direction * (START_GAP_M + distance), hold_fingers=True)
                for q in push:
                    reason = bare.reason(q) or bare.reason(self._command_pose(q))
                    if reason:
                        raise ValueError(f'push: {reason}')
                back = self._line(push[-1], -direction * BACKOFF_M, bare, hold_fingers=True)
                leave = self._line(back[-1], [0., 0., rise], bare)
                retract = tuple(back) + tuple(leave[1:])
                for name, path, w in (('enter', enter, world), ('retract', retract, bare)):
                    for q in path:
                        reason = w.reason(self._command_pose(q), **(policy if w is world else {}))
                        if reason:
                            raise ValueError(f'{name} under the gravity-compensated command: {reason}')
                failure = self._tracking_robust(enter, world, **policy)
                if failure:
                    raise ValueError(f'enter is not robust to arm tracking error: {failure}')
                open_path = self.edge(current, closed, world)
                approach, search = self.transport(closed, up[-1], world)
                phases = (Phase('open', open_path), Phase('approach', approach), Phase('enter', enter, jaw_contact=True),
                          PushPhase('nudge', tuple(push), drop_target=True), PushPhase('retract', retract, drop_target=True))
                final = cube.center_m + np.r_[direction[:2], 0.] * distance
                return ExtractionPlan(view.revision, target_id, phases, np.zeros((1, 3)), final, (search,))
            except ValueError as exc:
                rejections.append(str(exc))
        raise ValueError('no pusher pose works: ' + ('; '.join(sorted(set(rejections))[-3:]) or f'none reachable {dict(self.pusher_rejections)}'))

    # ------------------------------------------------------------------ choosing the push
    def _fits_in_box(self, view, xy, cube):
        inverse = np.linalg.inv(view.box.base_from_box)
        local = inverse[:3, :3] @ np.r_[xy, cube.center_m[2]] + inverse[:3, 3]
        r = cube.side_m * np.sqrt(.5) + .004
        return bool(np.all(local[:2] > view.box.inner_min_xy + r) and np.all(local[:2] < view.box.inner_max_xy - r))

    GRID_M = .01
    SEARCH_RADIUS_M = .09
    MIN_PUSH_M, MAX_PUSH_M = .004, .08
    PLACES = 6

    def graspable_places(self, view, target_id, current, log):
        """The closest places (1 cm grid in the box's own axes, faces square to the walls) where a grasp entry exists, nearest first."""
        cube = view.cubes[target_id]
        axes = [np.r_[view.box.base_from_box[:2, k] / np.linalg.norm(view.box.base_from_box[:2, k]), 0.] for k in (0, 1)]
        steps = np.arange(-self.SEARCH_RADIUS_M, self.SEARCH_RADIUS_M + 1e-9, self.GRID_M)
        candidates = sorted(((a, b) for a in steps for b in steps if np.hypot(a, b) <= self.SEARCH_RADIUS_M), key=lambda ab: np.hypot(*ab))
        found, tried = [], 0
        for a, b in candidates:
            xy = cube.center_m[:2] + a * axes[0][:2] + b * axes[1][:2]
            if not self._fits_in_box(view, xy, cube):
                continue
            if any(np.linalg.norm(other.center_m[:2] - xy) < cube.side_m * 1.8 for k, other in view.cubes.items() if k != target_id):
                continue
            tried += 1
            if self.entry_report(replace(view, cubes={**view.cubes, target_id: moved_cube(view, cube, xy)}), target_id, current)[0]:
                found.append((xy, axes))
                if len(found) >= self.PLACES:
                    break
        log(f'{target_id}: {len(found)} place(s) with a grasp entry within {self.SEARCH_RADIUS_M * 100:.0f} cm ({tried} places tried)')
        return found

    def plan_nudge(self, view, current, ids, log):
        """One push (plan, description) toward the nearest place where a cube can be grasped; the rest follows after a new measurement."""
        failures, feasible = [], {}
        for target_id in ids:
            cube = view.cubes[target_id]
            places = self.graspable_places(view, target_id, current, log)
            if not places:
                failures.append(f'{target_id}: no place near it admits a grasp')
                continue

            def drivable(direction):
                key = (target_id, tuple(np.round(direction[:2], 2)))
                if key not in feasible:
                    try:
                        feasible[key] = self.plan_push(view, current, target_id, direction, .01)
                    except ValueError as exc:
                        feasible[key] = None
                        failures.append(str(exc)[:300])
                        log(f'{target_id}: a push along {np.round(direction[:2], 2).tolist()} cannot be driven: {str(exc)[:300]}')
                return feasible[key] is not None
            for goal, axes in places:
                delta = goal - cube.center_m[:2]
                legs = sorted(((float(delta @ ax[:2]), ax) for ax in axes), key=lambda leg: -abs(leg[0]))
                legs = [(amount, ax * np.sign(amount)) for amount, ax in legs if abs(amount) >= self.MIN_PUSH_M]
                if not legs or not all(drivable(direction) for _, direction in legs):
                    continue
                amount, direction = legs[0]
                distance = min(abs(amount), self.MAX_PUSH_M)
                try:
                    plan = self.plan_push(view, current, target_id, direction, distance)
                except ValueError as exc:
                    failures.append(str(exc)[:300])
                    continue
                return plan, {'target': target_id, 'direction': direction[:2].tolist(), 'distance_m': distance, 'goal_xy_m': goal.tolist()}
        raise ValueError('no push moves a cube toward a place where it can be grasped' + (': ' + '; '.join(failures[-2:]) if failures else ''))
