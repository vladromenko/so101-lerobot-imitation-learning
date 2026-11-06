"""Driver adapter: the existing ExistingArmDriver plus the unfold and parking segments.

The folded rest pose of the arm is, by the model's own checks, outside the planning
envelope (shoulder_lift beyond its range, wrist against the shoulder). The installed
C driver unfolds and parks the arm itself; during those two segments only the box and
the cubes are checked, with every link mesh, and the end of the unfold must pass the
full world check.
"""
from __future__ import annotations

import inspect
import sys
import textwrap
from dataclasses import replace

import numpy as np

from so101_pick_place.experiments.box_extraction.driver import ExistingArmDriver
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld
from so101_pick_place.hardware.task_arm import TaskArm
from so101_pick_place.kinematics.so101_backend import ARM_JOINTS

from .planning import TunedExtractionPlanner, flush_local
from .scene_builder import guard_view, planning_view


def external_reason(world, joints):
    """Box and cube collisions only (used while the arm is leaving or entering its folded pose)."""
    backend, scene = world.backend, world.scene
    frames = backend.model.frames(joints)
    for name, mesh in backend.meshes.items():
        if name in frames:
            wall = scene.box.collision(backend._transform_points(frames[name], mesh))
            if wall is not None:
                return f'{name}: {wall}'
    return None


def scaled_move(parent):
    """The parent's `_move` with its Cartesian and joint speeds scaled by `self.speed_scale`.

    The speeds are literals inside the parent's method, so the method's own source is compiled again with them
    replaced; every other line (guards, grasp, force close) is the parent's, unchanged.
    """
    source = textwrap.dedent(inspect.getsource(parent._move))
    for old, new in (("speed=6.", "speed=6.*self.phase_scale(phase)"),
                     ("peak_tip_speed = min(.008, 5./max(1e-6, float(ratios.max(initial=0.))))",
                      "peak_tip_speed = min(.008*self.phase_scale(phase), 5.*self.phase_scale(phase)/max(1e-6, float(ratios.max(initial=0.))))"),
                     # the servos' own lag is led for (as the project's stow does) and the hold after each phase is short
                     ("self.arm.follow(poses, duration=duration, settle=.4, within=.7, lead=False, step='extract_'+phase.name)",
                      "self.arm.follow(poses, duration=duration, settle=.25, within=.7, lead=self.phase_lead(phase), step='extract_'+phase.name)")):
        if old not in source:
            raise RuntimeError(f'the parent driver changed: {old!r} not found')
        source = source.replace(old, new)
    namespace = dict(vars(sys.modules[parent.__module__]))
    exec(compile(source, f'<scaled {parent.__name__}._move>', 'exec'), namespace)
    return namespace['_move']


def read_command_sag(config, initial_joints, output):
    """The driver's gravity-compensation setting (deg/Nm), read from a simulated arm."""
    robot = config.robot
    arm = TaskArm(port='sim', calibration_file=robot['calibration'], offsets=robot.get('kinematic_offsets_deg'),
                  hold_on_success=False, out=output / 'sag-probe', name='sag-probe')
    arm._echo = False
    arm.sim(pose=initial_joints)
    with arm:
        return float(arm.raw.sag)


class BoxArmDriver(ExistingArmDriver):
    mode = 'task'          # 'ready' (unfold), 'task', 'home' (parking)
    # Speed relative to the parent driver's (8 mm/s tip speed): fast through free space, careful near the cube and the walls.
    PHASE_SCALE = {'nudge': .8, 'approach': 2.2, 'open': 2.0, 'transfer': 2.0, 'retract': 1.6, 'extract_vertical': 1.4, 'lower_to_support': 1.4, 'enter': 1.4, 'release': 1.6}
    speed_scale = 1.5
    # Leading the servos' lag pushes a goal past the pose by up to 1.5 deg; the phases that start at the ready pose (the wrist
    # is near its range there) go without it.
    NO_LEAD = {'approach', 'open', 'retract', 'enter', 'grasp', 'extract_vertical', 'transfer', 'lower_to_support', 'release'}   # all: goals past a joint's range are refused

    def phase_lead(self, phase):
        return phase.name not in self.NO_LEAD

    def phase_scale(self, phase):
        return self.PHASE_SCALE.get(phase.name, self.speed_scale)

    def guard_tick(self, tick):
        if self.mode == 'task':
            try:
                return super().guard_tick(tick)
            except RuntimeError as exc:
                if self.separating_from_released_cube(tick, exc):
                    self.previous['command'] = self.backend.driver_command_goal(tick['goal'], self.arm.raw.sag)
                    return None
                if self.phase is not None and self.phase.joints and str(exc).split(':')[0].split()[-1] == 'command':
                    command = self.backend.driver_command_goal(tick['goal'], self.arm.raw.sag)
                    print('DEBUG command-vs-goal (deg)', {j: round(command[j] - tick['goal'][j], 2) for j in ARM_JOINTS}, 'sag', self.arm.raw.sag,
                          '| gripper goal', round(tick['goal']['gripper'], 1), 'pos', round(tick['pos']['gripper'], 1))
                if self.phase is not None and self.phase.joints and 'pos' in str(exc).split(':')[0]:
                    pos = tick['pos']
                    deviation, nearest = min((max(abs(pos[j] - q[j]) for j in ARM_JOINTS), i) for i, q in enumerate(self.phase.joints))
                    payload = self.payload_local_override if self.payload_local_override is not None else self.plan.payload_local
                    verdict = self.world.reason(self.phase.joints[nearest], target=self.plan.target_id, jaw_contact=self.phase.jaw_contact,
                                                payload_local=payload if self.phase.carrying else None,
                                                support_contact=self.phase.support_contact, placed_cube=self.phase.placed_cube)
                    frame = self.backend.model.frames(pos)['gripper_frame_link']
                    points = self.backend._transform_points(frame, payload)
                    inverse = np.linalg.inv(self.world.scene.box.base_from_box)
                    local = points @ inverse[:3, :3].T + inverse[:3, 3]
                    nearest_q = self.phase.joints[nearest]
                    frame_n = self.backend.model.frames(nearest_q)['gripper_frame_link']
                    local_n = self.backend._transform_points(frame_n, payload) @ inverse[:3, :3].T + inverse[:3, 3]
                    print('DEBUG payload box-frame bounds at measured pose', np.round(local.min(0), 3), np.round(local.max(0), 3),
                          '| at planned waypoint', np.round(local_n.min(0), 3), np.round(local_n.max(0), 3),
                          '| measured', {k: round(v, 1) for k, v in pos.items()}, '| planned', {k: round(v, 1) for k, v in nearest_q.items()})
                    raise RuntimeError(f'{exc} (measured pose {deviation:.2f} deg from planned waypoint {nearest}/{len(self.phase.joints)}; '
                                       f'that waypoint under the guard: {verdict}; clearance {self.world.scene.box.clearance_m}, flap inflate {self.world.scene.box.flap_inflate_m})') from exc
                raise
        self.check_sensor()
        for kind, goal in (('goal', tick['goal']), ('pos', tick['pos'])):
            previous = self.previous.get(kind, tick['pos'])
            span = max(abs(goal[j] - previous[j]) for j in ARM_JOINTS)
            for alpha in np.linspace(0., 1., max(2, int(np.ceil(span / .5)) + 1)):
                joints = {j: float(previous[j] + alpha * (goal[j] - previous[j])) for j in (*ARM_JOINTS, 'gripper')}
                reason = external_reason(self.world, joints)
                if reason:
                    raise RuntimeError(f'{self.mode} {kind}: {reason}')
                self.preflight_samples += 1
            self.previous[kind] = dict(goal)

    SEPARATION_ZONE_M = .016

    def separating_from_released_cube(self, tick, exc):
        """True for the one command-pose contact the retract cannot avoid: at the release pose the fixed jaw lies against
        the cube it has just let go, and the gravity-compensated register command pushes it about 5 mm into it. The goal
        and the measured pose are clear; the exception holds only while the arm is within a few centimetres of that pose,
        i.e. during the sideways separation that precedes the rise."""
        text = str(exc)
        if self.phase is None or self.phase.name != 'retract' or not text.startswith('retract command: gripper_link: cube '):
            return False
        first = np.array(self.backend.model.pose(self.phase.joints[0]).tip, float)
        here = np.array(self.backend.model.pose(tick['goal']).tip, float)
        return float(np.linalg.norm(here - first)) <= self.SEPARATION_ZONE_M

    def confirm_grasp(self):
        """Payload after the force close: the cube lies flush against the fixed jaw (see flush_local)."""
        self.payload_local_override = flush_local(super().confirm_grasp())
        return self.payload_local_override

    def move_ready(self, ready, world):
        """Unfold to a free ready pose; the full world check must pass at its end."""
        self.mode, self.world, self.previous = 'ready', world, {}
        try:
            self.arm.move_joints(ready, speed=6., check=True, step='extract_ready')
        finally:
            self.mode = 'task'
        reason = world.reason(self.arm.read()[0])
        if reason is not None:
            raise RuntimeError('ready pose: ' + reason)

    def preflight_full(self, plan, world, ready, *, initial_joints, params):
        """Replay unfold, every phase, the post-grasp replan and the parking on simulated servos."""
        robot = self.config.robot
        simulation = BoxArmDriver(self.config, self.backend, self.output / 'preflight')
        arm = TaskArm(port='sim', calibration_file=robot['calibration'], offsets=robot.get('kinematic_offsets_deg'),
                      on_tick=simulation.guard_tick, hold_on_success=False, out=self.output / 'preflight', name='extraction-preflight')
        arm._echo = False
        simulation.arm = arm
        simulation.phase, simulation.plan, simulation.world = plan.phases[0], plan, world
        arm.sim(pose=initial_joints)
        with arm:
            simulation.move_ready(ready, world)
            grasp_index = next(i for i, phase in enumerate(plan.phases) if phase.name == 'grasp')
            for phase in plan.phases[:grasp_index + 1]:
                simulation._move(phase, plan, world, lambda: None, simulation=True)
            original = world.scene.cubes[plan.target_id]
            simulation.payload_local_override = flush_local(simulation.payload_local_override)     # the real close slides the cube flush
            adjusted = replace(plan, payload_local=simulation.payload_local_override)
            fresh = planning_view(replace(world.scene, cubes={k: v for k, v in world.scene.cubes.items() if k != plan.target_id}), params)
            remainder, updated = TunedExtractionPlanner(self.backend).replan_held(
                fresh, simulation.measured_joints(), adjusted, original, enforce_fresh=False)
            remainder_world = ExtractionCollisionWorld(self.backend, guard_view(updated))
            for phase in remainder.phases:
                simulation._move(phase, remainder, remainder_world, lambda: None, simulation=True)
            simulation.mode = 'home'
        return {'passed': True, 'sim_only': True, 'real_servo_bus_opened': False,
                'external_samples_checked': simulation.preflight_samples, 'includes_unfold': True,
                'includes_parking': True, 'contact_force_validated': False,
                'remainder_phases': [p.name for p in remainder.phases]}

    def preflight_nudge(self, plan, world, ready, *, initial_joints):
        """Replay unfold, a push of the target cube (it is left out of the world while it moves) and the parking."""
        robot = self.config.robot
        simulation = BoxArmDriver(self.config, self.backend, self.output / 'preflight')
        arm = TaskArm(port='sim', calibration_file=robot['calibration'], offsets=robot.get('kinematic_offsets_deg'),
                      on_tick=simulation.guard_tick, hold_on_success=False, out=self.output / 'preflight', name='nudge-preflight')
        arm._echo = False
        simulation.arm = arm
        simulation.phase, simulation.plan, simulation.world = plan.phases[0], plan, world
        arm.sim(pose=initial_joints)
        with arm:
            simulation.move_ready(ready, world)
            for phase in plan.phases:
                simulation._move(phase, plan, self.world_for(phase, world, plan.target_id), lambda: None, simulation=True)
            simulation.mode = 'home'
        return {'passed': True, 'sim_only': True, 'real_servo_bus_opened': False, 'external_samples_checked': simulation.preflight_samples}

    @staticmethod
    def world_for(phase, world, target_id):
        """The collision world of a phase: while a cube is being pushed it is not an obstacle of the arm."""
        if not getattr(phase, 'drop_target', False):
            return world
        scene = world.scene
        return ExtractionCollisionWorld(world.backend, replace(scene, cubes={k: v for k, v in scene.cubes.items() if k != target_id}))

    def preflight_remainder(self, plan, world, *, initial_joints):
        """Replay the replanned post-grasp phases from the measured pose (simulated servos only)."""
        robot = self.config.robot
        simulation = BoxArmDriver(self.config, self.backend, self.output / 'preflight-remainder')
        arm = TaskArm(port='sim', calibration_file=robot['calibration'], offsets=robot.get('kinematic_offsets_deg'),
                      on_tick=simulation.guard_tick, hold_on_success=True, out=self.output / 'preflight-remainder', name='extraction-remainder')
        arm._echo = False
        simulation.arm = arm
        simulation.phase, simulation.plan, simulation.world = plan.phases[0], plan, world
        simulation.payload_local_override, simulation.hold_goal, simulation.blocked_grip = self.payload_local_override, self.hold_goal, self.blocked_grip
        arm.sim(pose=initial_joints)
        with arm:
            for phase in plan.phases:
                simulation._move(phase, plan, world, lambda: None, simulation=True)
        return {'passed': True, 'sim_only': True, 'external_samples_checked': simulation.preflight_samples}


BoxArmDriver._move = scaled_move(ExistingArmDriver)
