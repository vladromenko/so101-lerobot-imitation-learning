"""Planner variant: identical phases, but the extraction lift is sized by the cube's lowest point."""
from dataclasses import replace
import time
import numpy as np
from scipy.optimize import least_squares
from so101_pick_place.experiments.box_extraction.planner import ExtractionPlanner, Phase, ExtractionPlan
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld
from so101_pick_place.geometry.spatial_cube import SpatialCube
from so101_pick_place.kinematics.so101_backend import ARM_JOINTS  # noqa: F401


def released_cube(placed, factor=1.):
    """Retain the measured cube size through release and retreat checks."""
    centre = placed.vertices_m.mean(0)
    return replace(placed, side_m=placed.side_m*factor, vertices_m=centre+(placed.vertices_m-centre)*factor)


def flush_local(local):
    """Cube vertices in the gripper frame after the force close.

    Closing slides the cube across the fixed jaw's stand-off until it lies against the fixed jaw plane (x = 0 in
    the gripper frame, the cube on its negative side): the held cube is that much closer than where it was measured.
    """
    local = np.asarray(local, float)
    return local - np.array([local[:, 0].max(), 0., 0.])


class TunedExtractionPlanner(ExtractionPlanner):
    timings = None

    def translated(self, *args, **kwargs):
        begin = time.perf_counter()
        try:
            return super().translated(*args,**kwargs)
        finally:
            if self.timings is not None:
                self.timings['ik_s'] = (self.timings.get('ik_s',0.) or 0.)+time.perf_counter()-begin

    def vertical_center(self, start, height, local_center, world, **policy):
        begin = time.perf_counter()
        collisions = (getattr(world,'metrics',{}).get('collision_checking_s',0.) or 0.)
        try:
            return super().vertical_center(start,height,local_center,world,**policy)
        finally:
            if self.timings is not None:
                collision_time = (getattr(world,'metrics',{}).get('collision_checking_s',0.) or 0.)-collisions
                self.timings['ik_s'] = (self.timings.get('ik_s',0.) or 0.)+max(0.,time.perf_counter()-begin-collision_time)

    def placement_pose(self, *args, **kwargs):
        begin = time.perf_counter()
        try:
            return super().placement_pose(*args,**kwargs)
        finally:
            if self.timings is not None:
                self.timings['ik_s'] = (self.timings.get('ik_s',0.) or 0.)+time.perf_counter()-begin
    rim_margin_m = .020          # lift clearance above the rim beyond the guard's own: a held cube hangs ~3 mm lower than planned
    fixed_jaw_gap_m = .0025      # clearance to the cube in the gravity-compensated command pose; the stand-off grows by the command shift per candidate
    opening_margin_m = .008      # jaw opening beyond the cube width (original: 14 mm); the box leaves little room

    command_sag = None           # deg/Nm of the driver's gravity compensation; set from the arm before planning
    max_standoff_m = .016
    release_separation_m = .010   # after release the fixed jaw moves this far from the cube before the retract rises

    def _jaw_angle_curve(self, reference, contact_z):
        samples = []
        for gripper in np.linspace(0., 45., 91):
            try:
                samples.append((self.backend._jaw_gap_at(reference, float(gripper), contact_z), float(gripper)))
            except Exception:
                continue
        samples.sort()
        gaps, angles = np.array([g for g, _ in samples]), np.array([a for _, a in samples])
        return lambda width: float(np.interp(width, gaps, angles))

    smoothing_deg = (.35, .15)            # rms distance a smoothed path may keep from the searched waypoints, tried in this order

    def transport(self, start, goal, world, **policy):
        """RRT-Connect's path, smoothed: the search returns a polyline whose every corner is a jerk when the arm drives it."""
        points, trace = super().transport(start, goal, world, **policy)
        return self._smooth(points, world, **policy), trace

    def _smooth(self, path, world, **policy):
        from scipy.interpolate import splev, splprep
        if len(path) < 5:
            return path
        q = np.array([[p[j] for j in ARM_JOINTS] for p in path], float)
        gripper = path[0]['gripper']
        chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        if chord[-1] < 1.:
            return path
        u = chord / chord[-1]
        keep = np.r_[True, np.diff(u) > 1e-9]
        for deg in self.smoothing_deg:
            try:
                tck, _ = splprep(q[keep].T, u=u[keep], s=int(keep.sum()) * deg ** 2, k=3)
            except Exception:
                continue
            count = max(len(path), int(chord[-1] / .5) + 1)
            dense = np.array(splev(np.linspace(0., 1., count), tck)).T
            dense[0], dense[-1] = q[0], q[-1]
            if np.any(dense < self.limits[:, 0]) or np.any(dense > self.limits[:, 1]):
                continue
            if np.abs(np.diff(dense, axis=0)).max() > 2.5:
                continue
            if all(world.reason(self.joints(point, gripper), **policy) is None for point in dense[::2]) and world.reason(self.joints(dense[-1], gripper), **policy) is None:
                return tuple(self.joints(point, gripper) for point in dense)
        return path

    release_openings = (45., 38., 32., 27.)

    def _release(self, start, world, first=45., **policy):
        """Open the jaws at the placement: as wide as the walls around the spot allow (widest first), but wider than the cube."""
        reason = None
        for opening in [o for o in self.release_openings if o <= first + 1e-6] or [first]:
            goal = dict(start, gripper=opening)
            try:
                return goal, self.edge(start, goal, world, **policy)
            except ValueError as exc:
                reason = exc
        raise reason

    def _retract_path(self, release_goal, center_local, world, **policy):
        """Fixed jaw away from the released cube first (it is flush against it), then straight up."""
        frame = self.model.frames(release_goal)['gripper_frame_link']
        away = self.translated(release_goal, frame[:3, 0]*self.release_separation_m)
        first = self.edge(release_goal, away, world, **policy)
        up = self.vertical_center(away, .03, center_local, world, **policy)
        self.separation_waypoints = len(first)
        return tuple(first)+tuple(up)[1:]

    separation_waypoints = 0
    grasp_pitches_deg = (-90., -70., -50.)      # tool pitch seeds of the grasp search
    min_tool_down = .5                           # fingers' axis must point at least this far down (sine of the angle below horizontal)
    tracking_error_deg = 3.0      # the arm trails its path by about this much; the entry and closure must survive it

    @staticmethod
    def _intended(reason, target_id):
        """The contacts the driver itself allows during the jaw closure: the two fingers on the target cube."""
        return reason in {f'gripper_link: cube {target_id}', f'moving_jaw_so101_v1_link: cube {target_id}'}

    def _lagging_poses(self, path):
        """Poses the arm can be in while it trails its path: every joint behind its waypoint in proportion to its own
        speed (the fastest by `tracking_error_deg`), with different gains per joint, so the pose is off the path."""
        patterns = [(1.,)*5] + [tuple(1.6 if i == j else .6 for i in range(5)) for j in range(5)] \
                             + [tuple(.2 if i == j else 1. for i in range(5)) for j in range(5)]
        for k in range(1, len(path)-1, 2):
            velocity = np.array([(path[k+1][j]-path[k-1][j])/2. for j in ARM_JOINTS])
            peak = float(np.abs(velocity).max())
            if peak < 1e-6:
                continue
            behind = velocity*(self.tracking_error_deg/peak)
            for gains in patterns:
                yield dict(path[k], **{j: path[k][j]-float(g*d) for j, g, d in zip(ARM_JOINTS, gains, behind)})

    def _tracking_robust(self, path, world, *, closing=False, **policy):
        """The jaws stay clear of the cube and the walls while the arm trails its path."""
        for pose in self._lagging_poses(path):
            reason = world.reason(pose, **policy)
            # Overshooting into the table is not a lag direction (the arm trails its goal from above); the guard still stops it.
            if reason and 'table' not in reason and 'floor' not in reason and not (closing and self._intended(reason, policy['target'])):
                return f'trailing the path by up to {self.tracking_error_deg:.1f} deg: {reason}'
        return None

    def _command_pose(self, joints):
        return self.backend.driver_command_goal(joints, self.command_sag) if self.command_sag else dict(joints)

    def _fixed_jaw_clearance(self, joints, cube):
        """Distance from the fixed-jaw plane to the nearest cube vertex along the closing axis."""
        frame = self.model.frames(joints)['gripper_frame_link']
        return float(-((cube.vertices_m - frame[:3, 3]) @ frame[:3, :3])[:, 0].max())

    def grasp_candidates(self, scene, cube, current):
        """Solve opposing measured cube faces against the real jaw frame.

        The fixed jaw's stand-off is sized per candidate so that the gravity-compensated register command
        (the arm that is actually commanded) still leaves `fixed_jaw_gap_m` to the cube; the jaw opening
        and the expected contact angle follow from that stand-off.
        """
        reference = self.model.pose(current)
        # Jaw contact height leaves the real distal mesh above the floor.
        floor_world = scene.box.base_from_box @ np.r_[0., 0., scene.box.floor_interval_m[1], 1.]
        contact_z = self.backend.fixed_jaw_distal_z_m + floor_world[2]+self.model.gap+.003-cube.center_m[2]
        for _ in range(3):
            opening, _ = self.backend.expected_block_gripper(reference, cube.side_m+self.opening_margin_m, contact_z)
            closed, _ = self.backend.expected_block_gripper(reference, cube.side_m, contact_z)
            distal = self.backend.fixed_jaw_distal_z_m
            for gripper in np.linspace(closed, opening, 12):
                frames = self.model.frames(dict(current, gripper=float(gripper)))
                moving = self.backend._transform_points(np.linalg.inv(frames['gripper_frame_link']) @ frames['moving_jaw_so101_v1_link'], self.backend.meshes['moving_jaw_so101_v1_link'])
                distal = max(distal, float(moving[:, 2].max()))
            contact_z = distal+floor_world[2]+max(self.model.gap, scene.box.clearance_m)+.003-cube.center_m[2]
        if not 4. <= closed < opening <= 45.:
            raise ValueError('measured cube does not fit the reliable jaw range')
        angle_for = self._jaw_angle_curve(reference, contact_z)
        clamp = angle_for(cube.side_m)
        seen = set()
        for axis in cube.rotation.T:
            for sign in (-1., 1.):
                closing = sign*axis
                if abs(closing[2]) < .65:
                    frame_x = -closing
                    tool_z = np.array([0., 0., -1.])-frame_x*(-frame_x[2])
                    tool_z /= np.linalg.norm(tool_z)
                    desired = np.column_stack((frame_x, np.cross(tool_z, frame_x), tool_z))
                    seeds = [(pitch, roll) for pitch in self.grasp_pitches_deg for roll in np.linspace(*self.limits[4], 9)]

                    def fit(seed_pose, gap):
                        local = np.array([-(cube.side_m/2.+gap), 0., contact_z])
                        def residual(q):
                            frame = self.model.frames(self.joints(q, opening))['gripper_frame_link']
                            return np.r_[(frame[:3, 3]+frame[:3, :3]@local-cube.center_m)*1000., (frame[:3, 0]-frame_x)*30.]
                        solve_started = time.perf_counter()
                        try:
                            result = least_squares(residual, self.vector(seed_pose.deg), bounds=(self.limits[:, 0], self.limits[:, 1]), max_nfev=80)
                        finally:
                            if self.timings is not None:
                                self.timings['ik_s'] = (self.timings.get('ik_s',0.) or 0.)+time.perf_counter()-solve_started
                        error = residual(result.x)
                        joints = self.joints(result.x, opening)
                        actual = self.model.frames(joints)['gripper_frame_link']
                        good = (np.all(np.isfinite(result.x)) and np.linalg.norm(error[:3]) <= 1.5
                                and np.linalg.norm(error[3:6])/30. <= np.sin(np.deg2rad(2.)) and actual[2, 2] < -self.min_tool_down)
                        return (joints if good else None), result.x
                    for pitch, roll in seeds:
                        target_tip = cube.center_m-desired@np.array([-(cube.side_m/2.+self.fixed_jaw_gap_m), 0., contact_z])
                        solve_started = time.perf_counter()
                        try:
                            seed = self.model.ik(*target_tip, pitch=pitch, roll=float(roll), gripper=opening)
                        except self.backend._unreachable:
                            seed = None
                        finally:
                            if self.timings is not None:
                                self.timings['ik_s'] = (self.timings.get('ik_s',0.) or 0.)+time.perf_counter()-solve_started
                        if seed is None:
                            continue
                        joints, x = fit(seed, self.fixed_jaw_gap_m)
                        if joints is None:
                            continue
                        # How much closer the command pose brings the fixed jaw than the goal pose does.
                        cmd = self._command_pose(joints)
                        gap = self.fixed_jaw_gap_m + max(0., self._fixed_jaw_clearance(joints, cube) - self._fixed_jaw_clearance(cmd, cube)) + .001
                        if gap > self.max_standoff_m:
                            continue
                        if gap > self.fixed_jaw_gap_m + .0005:
                            joints, x = fit(self.model.pose(joints), gap)
                            if joints is None:
                                continue
                            cmd = self._command_pose(joints)
                        if self._fixed_jaw_clearance(cmd, cube) < .0015:
                            continue
                        open_angle = angle_for(cube.side_m+gap+self.opening_margin_m)
                        if not angle_for(cube.side_m+gap) < open_angle <= 45.:
                            continue
                        joints = dict(joints, gripper=open_angle)
                        signature = tuple(np.round(self.vector(joints), 1))
                        if signature not in seen:
                            seen.add(signature)
                            self.last_clamp_deg = clamp
                            yield joints, angle_for(cube.side_m+gap)

    def plan(self, scene, current):
        """The parent's search over the cubes in the box, but never for an object that could not be measured."""
        real = {k: c for k, c in scene.cubes.items() if not c.evidence.get('unmeasured')}
        targets = replace(scene, cubes=real)
        if not real:
            raise ValueError('no measured cubes remain inside the box')
        world = ExtractionCollisionWorld(self.backend, scene)          # every object, measured or not, is an obstacle
        scene.require_complete()
        inverse = np.linalg.inv(scene.box.base_from_box)
        rejections = []
        for target_id, cube in sorted(targets.cubes.items(), key=lambda item: -item[1].center_m[2]):
            local = inverse[:3, :3] @ cube.center_m + inverse[:3, 3]
            inside = np.all(local[:2] > scene.box.inner_min_xy) and np.all(local[:2] < scene.box.inner_max_xy) and scene.box.floor_interval_m[0] < local[2] < scene.box.obstacle_top_m
            if not inside:
                continue
            try:
                for grasp, closed in self.grasp_candidates(scene, cube, current):
                    try:
                        return self._candidate(scene, current, target_id, cube, grasp, closed, world)
                    except ValueError as exc:
                        rejections.append(str(exc))
            except ValueError as exc:
                rejections.append(str(exc))
        raise ValueError('no collision-free measured grasp and placement: ' + ('; '.join(rejections[-3:]) or 'IK/jaw geometry rejected all face pairs'))

    def _candidate(self, scene, current, target_id, cube, grasp, closed, world):
        policy = {'target': target_id, 'jaw_contact': True}
        frame = self.model.frames(grasp)['gripper_frame_link']
        local = (cube.vertices_m-frame[:3, 3]) @ frame[:3, :3]
        rim = (scene.box.base_from_box @ np.r_[0., 0., scene.box.obstacle_top_m, 1.])[2]
        radius = float(np.linalg.norm(cube.vertices_m-cube.center_m, axis=1).max())
        # The cube only has to clear the rim: its lowest point, not its corner radius, sets the lift.
        rise = max(.025, rim+scene.box.clearance_m+self.rim_margin_m+cube.side_m/2-cube.center_m[2])
        entry_up = self.vertical_center(grasp, rise, local.mean(0), world, **policy)
        approach = entry_up[-1]
        opened = dict(current, gripper=grasp['gripper'])
        open_path = self.edge(current, opened, world)
        approach_path, search1 = self.transport(opened, approach, world)
        # The existing contact-band estimate initializes the aperture. Refine
        # first contact against the original, concave moving-jaw triangles;
        # never ask a geometric path to pass through the measured cube.
        contact = None
        previous = grasp['gripper']
        for opening in np.arange(grasp['gripper'], max(4., closed-5.), -.25):
            reason = world.reason(dict(grasp, gripper=float(opening)), **policy)
            if reason is not None:
                if reason == f'moving_jaw_so101_v1_link: cube {target_id}':
                    contact = previous
                    break
                raise ValueError('grasp closure: '+reason)
            previous = float(opening)
        if contact is None or abs(contact-closed) > 5.:
            raise ValueError('full jaw mesh does not confirm the expected contact interval')
        closed = contact
        closure = self.edge(grasp, dict(grasp, gripper=closed), world, **policy)
        # Closing slides the cube across the stand-off onto the fixed jaw, so the jaws end at the width of the cube: the
        # clamped angle is what the driver expects from the force close; the path itself stops at first contact.
        held = dict(closure[-1], gripper=min(closure[-1]['gripper'], self.last_clamp_deg))
        closure = [*closure, held]
        flush = flush_local(local)
        lift = self.vertical_center(held, rise, flush.mean(0), world, payload_local=flush, **policy)
        # Select a point from the measured support polygon, then verify the
        # complete cube footprint; no nominal table coordinates are used.
        patch = scene.placement
        center = np.r_[np.asarray(patch.polygon_xy_m).mean(0), patch.height_m+patch.uncertainty_m+cube.side_m/2+.001]
        down = self.placement_pose(held, flush, cube, center)
        down_frame = self.model.frames(down)['gripper_frame_link']
        placed_vertices = self.backend._transform_points(down_frame, flush)
        if not patch.contains(placed_vertices):
            raise ValueError('measured free table patch cannot contain this cube and clearance')
        placement_up = self.vertical_center(down, rise, flush.mean(0), world, payload_local=flush, support_contact=True, **policy)
        above = placement_up[-1]
        transfer, search2 = self.transport(lift[-1], above, world, payload_local=flush, **policy)
        lowering = placement_up[::-1]
        # Open to the validated reliable limit before retracting. Keeping only
        # the approach aperture left the moving fingertip beside the placed
        # cube, where driver compensation could brush it during the first
        # upward millimetres.
        release_goal, release = self._release(lowering[-1], world, payload_local=flush, support_contact=True, **policy)
        placed = replace(cube, center_m=placed_vertices.mean(0), vertices_m=placed_vertices,
                         rotation=down_frame[:3, :3] @ frame[:3, :3].T @ cube.rotation)
        placed = released_cube(placed)
        retract = self._retract_path(release_goal, flush.mean(0), world, placed_cube=placed, **policy)
        if self.command_sag:
            for name, path, extra in (('enter', entry_up, {}), ('closure', closure[:-1], {}), ('lift', lift, {'payload_local': flush}),
                                      ('placement', placement_up, {'payload_local': flush, 'support_contact': True}),
                                      ('retract', retract, {'placed_cube': placed})):
                for q in (path[self.separation_waypoints:] if name == 'retract' else path):
                    reason = world.reason(self._command_pose(q), **(policy | extra))
                    if reason and not (name == 'closure' and self._intended(reason, target_id)):
                        raise ValueError(f'{name} under the gravity-compensated command: {reason}')
        for name, path, closing in (('enter', entry_up, False), ('closure', closure[:-1], True)):
            failure = self._tracking_robust(path, world, closing=closing, **policy)
            if failure:
                raise ValueError(f'{name} is not robust to arm tracking error: {failure}')
        phases = (Phase('open', open_path), Phase('approach', approach_path),
                  Phase('enter', entry_up[::-1], jaw_contact=True),
                  Phase('grasp', closure, jaw_contact=True),
                  Phase('extract_vertical', lift, True, True),
                  Phase('transfer', transfer, True, True),
                  Phase('lower_to_support', lowering, True, True, True),
                  Phase('release', release, True, True, True),
                  Phase('retract', retract, jaw_contact=True, placed_cube=placed))
        return ExtractionPlan(scene.revision, target_id, phases, flush, center, (search1, search2))

    def replan_held(self, scene, current, previous_plan, original_cube, *, enforce_fresh=True):
        """Replan from measured feedback without another grasp or release.

        The perception layer must exclude the positively identified held cube
        from the static map; its identity is reserved across fresh surveys.
        """
        scene.require_complete(now=None if enforce_fresh else scene.captured_at)
        target_id = previous_plan.target_id
        if target_id in scene.cubes:
            raise ValueError('held-object identity collides with a newly measured static obstacle')
        frame = self.model.frames(current)['gripper_frame_link']
        local = previous_plan.payload_local
        vertices = self.backend._transform_points(frame, local)
        original_centered = original_cube.vertices_m-original_cube.center_m
        local_rotation = np.linalg.lstsq(original_centered, local-local.mean(0), rcond=None)[0].T @ original_cube.rotation
        cube = SpatialCube(vertices.mean(0), original_cube.side_m, frame[:3, :3] @ local_rotation,
                           vertices, original_cube.evidence | {'geometry_source': 'measured FK and force-confirmed rigid attachment'})
        updated = replace(scene, cubes=scene.cubes | {target_id: cube})
        world = ExtractionCollisionWorld(self.backend, updated)
        policy = {'target': target_id, 'jaw_contact': True, 'payload_local': local}
        rim = (updated.box.base_from_box @ np.r_[0., 0., updated.box.obstacle_top_m, 1.])[2]
        radius = float(np.linalg.norm(vertices-cube.center_m, axis=1).max())
        rise = max(0., rim+updated.box.clearance_m+self.rim_margin_m+cube.side_m/2-cube.center_m[2])
        lift = self.vertical_center(current, rise, local.mean(0), world, **policy) if rise > .001 else (current,)
        patch = updated.placement
        center = np.r_[np.asarray(patch.polygon_xy_m).mean(0), patch.height_m+patch.uncertainty_m+cube.side_m/2+.001]
        down = self.placement_pose(current, local, cube, center)
        placement_frame = self.model.frames(down)['gripper_frame_link']
        placed_vertices = self.backend._transform_points(placement_frame, local)
        if not patch.contains(placed_vertices):
            raise ValueError('newly measured support cannot contain the held cube')
        descent_height = max(.025, rim+updated.box.clearance_m+self.rim_margin_m+cube.side_m/2-placed_vertices.mean(0)[2])
        placement_up = self.vertical_center(down, descent_height, local.mean(0), world, support_contact=True, **policy)
        above = placement_up[-1]
        transfer, search = self.transport(lift[-1], above, world, **policy)
        lower = placement_up[::-1]
        opening = next(p.joints[-1]['gripper'] for p in previous_plan.phases if p.name == 'release')
        release_goal, release = self._release(lower[-1], world, first=opening, support_contact=True, **policy)
        placed = replace(cube, center_m=placed_vertices.mean(0), vertices_m=placed_vertices,
                         rotation=placement_frame[:3, :3] @ local_rotation)
        placed = released_cube(placed)
        retract = self._retract_path(release[-1], local.mean(0), world, target=target_id, jaw_contact=True, placed_cube=placed)
        if self.command_sag:
            for name, path, extra in (('lift', lift, {}), ('placement', placement_up, {'support_contact': True}),
                                      ('retract', retract, {'payload_local': None, 'placed_cube': placed})):
                for q in (path[self.separation_waypoints:] if name == 'retract' else path):
                    reason = world.reason(self._command_pose(q), **(policy | extra))
                    if reason:
                        raise ValueError(f'{name} under the gravity-compensated command: {reason}')
        phases = (Phase('extract_vertical', lift, True, True), Phase('transfer', transfer, True, True),
                  Phase('lower_to_support', lower, True, True, True), Phase('release', release, True, True, True),
                  Phase('retract', retract, jaw_contact=True, placed_cube=placed))
        return ExtractionPlan(updated.revision, target_id, phases, local, center, (search,)), updated
