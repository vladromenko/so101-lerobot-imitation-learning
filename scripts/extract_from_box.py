#!/usr/bin/env python
"""Take cubes out of the open box with the SO-101, using the overview camera and SAM 3.

    extract_from_box.py                       # measure + plan + simulate; never touches the motors
    extract_from_box.py --execute --confirm-attended [--max-cubes N]

Every cycle: three agreeing SAM 3 scans -> metric box/cubes/free table patch -> mesh-accurate
extraction plan (grasp, vertical lift above the rim, transfer, gentle placement) -> replay of the
complete motion incl. unfold, post-grasp replan and parking on the installed driver's simulated
servos -> fresh scan identical to the plan -> execution with a fresh scan before every phase ->
parking -> scan confirming the cube is on the table and out of the box.
Stop: Ctrl+C or the STOP button of the existing web UI (the arm holds its pose).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('SO101_PROJECT', Path.home() / 'Projects/so101_autonomous_pick_place'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'scripts'))

from so101_pick_place.config import load_config                                         # noqa: E402
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld   # noqa: E402
from so101_pick_place.experiments.box_extraction.planner import Phase                    # noqa: E402
from so101_pick_place.kinematics.so101_backend import ARM_JOINTS                          # noqa: E402
from so101_pick_place.hardware.camera_watchdog import CameraWatchdog                     # noqa: E402
from so101_pick_place.hardware.observation_feedback import wait_stable_feedback          # noqa: E402
from so101_pick_place.hardware.robot import ReadOnlySO101                                # noqa: E402
from so101_pick_place.hardware.task_arm import TaskArm                                   # noqa: E402
from so101_pick_place.kinematics import So101ModelPlanner                                # noqa: E402
from so101_pick_place.telemetry.service import TickPublisher, atomic_record              # noqa: E402

from box_extract.driver import BoxArmDriver, read_command_sag                                              # noqa: E402
from box_extract.perception import OverviewPerception, SceneChanged                      # noqa: E402
from box_extract.planning import TunedExtractionPlanner
from box_extract.nudge import NudgePlanner, PushPhase                     # noqa: E402
from box_extract.stow import StowPlanner                                  # noqa: E402
from box_extract.shift import MAX_SHIFT_M, correct_pending, shift_scene  # noqa: E402
from box_extract.scene_builder import SceneError, SceneParameters, inside_box, projection, guard_view, planning_view  # noqa: E402

API = 'http://127.0.0.1:8010'


class Restart(SceneChanged):
    """The arm backed out to its ready pose after a failed attempt; measure and plan again."""


class Infeasible(RuntimeError):
    """The scene is valid but no collision-free extraction exists."""


def stamp():
    return datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')


def choose_ready(backend, scene, gripper=35.):
    """Free, reachable, unfolded start pose above the table, derived by IK (no saved angles)."""
    world = ExtractionCollisionWorld(backend, scene)
    for z in (.15, .13, .11):
        for x, y in ((.16, -.05), (.18, 0.), (.14, -.05), (.20, -.05), (.16, .05)):
            for pitch in (-90., -75., -60.):
                try:
                    pose = backend.model.ik(x, y, z, pitch=pitch, roll=0., gripper=gripper).deg
                except Exception:
                    continue
                if world.reason(pose) is None:
                    return {k: float(v) for k, v in pose.items()}
    raise Infeasible('no free reachable ready pose above the table')


def plan_scene(planner, scene, patches, ready, log, params, *, max_patches=6):
    failures = []
    for patch in patches[:max_patches]:
        trial = replace(scene, placement=patch, captured_at=time.time())
        started = time.time()
        try:
            plan = planner.plan(planning_view(trial, params), ready)
            log(f'plan found for patch {np.round(patch.evidence["center_xy_m"], 3).tolist()} in {time.time() - started:.0f} s')
            return plan, trial
        except Exception as exc:                                  # noqa: BLE001 - reported, never hidden
            failures.append(f'patch {np.round(patch.evidence["center_xy_m"], 3).tolist()}: {exc}')
            log(failures[-1][:300])
    raise Infeasible('; '.join(failures[-3:]) or ('there is no room left on the box floor for another cube (it holds as many as fit with the gripper\'s clearance)' if getattr(planner, 'stowing', False) else 'no free table patch for the extracted cube'))


# (margin around the flaps, wall clearance for the planned paths), widest first. A held cube trails its path by up to
# ~1 cm in the real arm, so the wide rows are tried first and narrowed only when no path exists; whatever row is used,
# the simulated replay with the true geometry and the guard during motion still have to pass.
MAX_RESTARTS = 4
# A fresh scan before every phase cost ~8 s each with the arm standing still; the scene can only matter before these.
CHECKED_PHASES = {'open', 'approach', 'enter', 'grasp', 'transfer'}
MAX_PLAN_ATTEMPTS = 3
MAX_NUDGES = 5            # pushes of a cube toward a place where it can be grasped, per cube taken out
MARGIN_LADDER_M = ((.024, .012), (.016, .010), (.010, .008), (.005, .006), (.005, .004))


def plan_relaxing(planner, scene, patches, ready, log, params):
    """Plan with the widest margins that leave a path, narrowing them only when none exists."""
    last = None
    for flap, wall in MARGIN_LADDER_M:
        trial_params = replace(params, flap_planning_margin_m=flap, planning_wall_clearance_m=wall)
        try:
            plan, trial = plan_scene(planner, scene, patches, ready, log, trial_params, max_patches=2)
        except Infeasible as exc:
            last = exc
            log(f'no path with {flap * 1000:.0f} mm around the flaps and {wall * 1000:.0f} mm to the walls')
            continue
        log(f'planned with {flap * 1000:.0f} mm around the flaps and {wall * 1000:.0f} mm to the walls')
        return plan, trial, trial_params
    raise last or Infeasible('no margins to plan with')


def plan_record(plan, backend):
    """plan.public() without the jaw-centre trace, which needs a contact band that open-jaw waypoints lack."""
    model = backend.model
    return {'scene_revision': plan.scene_revision, 'target_id': plan.target_id,
            'placement_center_m': plan.placement_center_m.tolist(),
            'phases': [{'name': p.name, 'carrying': p.carrying, 'joints': list(p.joints),
                        'path_tip_m': [list(model.pose(q).tip) for q in p.joints]} for p in plan.phases],
            'searches': list(plan.searches)}


def write_overlay(path, frame, scene, plan, calibration):
    proj = projection(calibration, 0.)
    image = frame.copy()
    for key, cube in scene.cubes.items():
        pts = np.array([proj.table_to_pixel(tuple(v)) for v in cube.vertices_m[4:]], np.int32)
        color = (0, 255, 255) if plan is not None and key == plan.target_id else (0, 0, 255)
        cv2.polylines(image, [pts], True, color, 2)
        cv2.putText(image, key, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    corners = np.array([[x, y, 0.] for x in (scene.box.inner_min_xy[0], scene.box.inner_max_xy[0]) for y in (scene.box.inner_min_xy[1], scene.box.inner_max_xy[1])])
    corners = np.array([scene.box.base_from_box[:3, :3] @ c + scene.box.base_from_box[:3, 3] for c in corners])
    pts = np.array([proj.table_to_pixel(tuple(c)) for c in corners[[0, 1, 3, 2]]], np.int32)
    cv2.polylines(image, [pts], True, (0, 255, 0), 2)
    if plan is not None:
        patch = np.array([proj.table_to_pixel((*p, 0.)) for p in scene.placement.polygon_xy_m], np.int32)
        cv2.polylines(image, [patch], True, (255, 0, 255), 2)
        cv2.putText(image, 'place', tuple(patch[0]), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)
    cv2.imwrite(str(path), image)


def require_attended(args):
    from deterministic_pick_place import require_approval
    approval = require_approval()
    attended = json.loads((ROOT / 'runtime/attended_session.json').read_text())
    if attended.get('operator_ready') is not True or attended.get('operator_presence_recheck_pending') is True:
        raise RuntimeError('current attended STOP readiness is required (runtime/attended_session.json)')
    if not args.confirm_attended:
        raise RuntimeError('--confirm-attended is required: the operator must be present with STOP at hand')
    return approval


def execute(args, config, calibration, backend, perception, planner, params, scene, plan, ready, initial, output, log, *, base_scene, patches):
    from deterministic_pick_place import publish_stage
    lease = ROOT / 'runtime/single_cube_executor.json'
    nonce, owned = uuid.uuid4().hex, False
    if lease.exists():
        raise RuntimeError('another executor owns the servo bus (runtime/single_cube_executor.json)')
    descriptor = os.open(lease, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    owned = True
    with os.fdopen(descriptor, 'w') as handle:
        json.dump({'pid': os.getpid(), 'script': Path(__file__).name, 'owner_nonce': nonce, 'artifacts': str(output)}, handle)
    try:
        publish_stage('SCANNING')
        telemetry, last = TickPublisher(ROOT, purpose='box_extraction'), [0.]
        state = {'driver': None, 'arm': None}

        def tick(record):
            cameras.check()
            state['driver'].guard_tick(record)
            if time.monotonic() - last[0] >= .1:
                publish_stage(state['stage'], joints=record['pos'], paused_for_vision=False)
                telemetry.tick(record, state['arm'].raw.sag, holding=state['driver'].retention.armed)
                last[0] = time.monotonic()
        state['stage'] = 'APPROACHING'
        robot = config.robot
        with CameraWatchdog(API, required=('overview',)) as cameras:
            arm = TaskArm(port=robot['port'], calibration_file=robot['calibration'], offsets=robot.get('kinematic_offsets_deg'),
                          on_tick=tick, hold_on_success=False, out=output / 'controller', name='box-extraction')
            driver = BoxArmDriver(config, backend, output, arm=arm, check_sensor=cameras.check)
            if abs(arm.raw.sag - planner.command_sag) > 1e-9:
                raise RuntimeError('the live arm uses a different gravity-compensation setting than the planned commands')
            state['arm'], state['driver'] = arm, driver
            current_plan, current_scene = plan, scene
            world = ExtractionCollisionWorld(backend, scene)
            driver.phase, driver.plan, driver.world = plan.phases[0], plan, world
            original = scene.cubes[plan.target_id]
            carrying = False
            executed = []

            def retrace_to_ready(reason):
                """Open the jaws and back out along the way in (the phases already driven, reversed), then ask for a fresh start."""
                log(f'{reason}: opening the jaws and backing out along the way in')
                forward = [q for done in executed if done.name in ('approach', 'enter') for q in done.joints]
                actual = driver.measured_joints()
                if forward:
                    deviation = [max(abs(q[j] - actual[j]) for j in ARM_JOINTS) for q in forward]
                    nearest = int(np.argmin(deviation))
                    if deviation[nearest] > 4.:
                        raise RuntimeError(f'{reason}; the arm is {deviation[nearest]:.1f} deg off the way in, holding instead of backing out')
                    opening = float(forward[nearest]['gripper'])
                    state['stage'] = 'RETREATING'
                    if abs(actual['gripper'] - opening) > 1.:
                        driver.move_phase(Phase('open', (dict(actual), dict(actual, gripper=opening)), jaw_contact=True), current_plan, world, lambda: None)
                    back = tuple(dict(forward[k], gripper=opening) for k in range(nearest, -1, -1))
                    driver.move_phase(Phase('retreat', back, jaw_contact=True), current_plan, world, lambda: None)
                raise Restart(reason)
            with arm:
                measured, _, _ = arm.read()
                if max(abs(measured[j] - initial[j]) for j in initial if j != 'gripper') > 1.:
                    raise RuntimeError('the arm moved after the planning measurement; holding')
                publish_stage('APPROACHING', joints=measured)
                driver.move_ready(ready, world)
                queue = list(plan.phases)
                while queue:
                    phase = queue.pop(0)
                    joints, _, _ = wait_stable_feedback(arm, cameras.check)
                    state['stage'] = phase.name.upper()
                    publish_stage(state['stage'], joints=joints, paused_for_vision=True)
                    if carrying:
                        driver.confirm_retention()
                    if phase.name in CHECKED_PHASES:
                        try:
                            check = perception.assert_unchanged(
                                current_scene, target_id=plan.target_id, joints=joints, carrying=carrying,
                                arm_over_box=phase.name in {'approach', 'enter', 'grasp', 'extract_vertical'})
                        except SceneChanged as changed:
                            if phase.name in ('approach', 'enter', 'grasp') and not carrying:
                                retrace_to_ready(f'{changed}')
                            changed.phase = phase.name
                            driver.hold()
                            raise
                        log(f'{phase.name}: scene unchanged {check}')
                        shift = check['box_shift_xy_m']
                        if shift is not None and any(isinstance(p, PushPhase) for p in plan.phases):
                            retrace_to_ready(f'the box moved {float(np.linalg.norm(shift)) * 1000:.1f} mm; a push is planned for the old place')
                        if shift is not None:
                            size = float(np.linalg.norm(shift))
                            if carrying:
                                if size > .004:
                                    driver.hold()
                                    raise SceneChanged(f'the box moved {size * 1000:.1f} mm while the cube was in the gripper')
                            else:
                                moved_scene = shift_scene(current_scene, shift)
                                moved_world = ExtractionCollisionWorld(backend, moved_scene)
                                base_scene = shift_scene(base_scene, shift)
                                try:
                                    pending = correct_pending(planner, moved_world, [phase, *queue], shift, plan.target_id)
                                except ValueError as exc:
                                    # Too far for the same path to follow: plan again from where the arm is, then replay it on
                                    # simulated servos exactly as before the first motion.
                                    log(f'the box moved {size * 1000:.1f} mm; the same path cannot follow it ({exc}); planning again from the current pose')
                                    measured = driver.measured_joints()
                                    try:
                                        new_plan, moved_scene, params = plan_relaxing(planner, replace(base_scene, captured_at=time.time()), patches, measured, log, params)
                                        moved_world = ExtractionCollisionWorld(backend, moved_scene)
                                        driver.preflight_remainder(new_plan, moved_world, initial_joints=measured)
                                    except Exception as failure:
                                        driver.hold()
                                        raise SceneChanged(f'the box moved {size * 1000:.1f} mm and no path to the cube exists: {failure}') from failure
                                    plan, current_plan = new_plan, new_plan
                                    pending = list(new_plan.phases)
                                phase, *queue = pending
                                current_scene, world = moved_scene, moved_world
                                original = moved_scene.cubes[plan.target_id]
                                perception.accept_box_shift(check['box_shift_px_vec'])
                                log(f'the box moved {size * 1000:.1f} mm ({shift[0] * 1000:+.1f}, {shift[1] * 1000:+.1f}): the remaining path follows it')
                    log(f'phase {phase.name}: {len(phase.joints)} waypoints')
                    try:
                        driver.move_phase(phase, current_plan, driver.world_for(phase, world, plan.target_id), lambda: None)
                    except RuntimeError as failure:
                        # A grasp that closes on nothing (the cube was not where the jaws went) leaves the arm unharmed in the box.
                        if phase.name == 'grasp' and 'measured grasp differs' in str(failure) and driver.measured_joints()['gripper'] < 5.:
                            retrace_to_ready('the jaws closed on nothing')
                        raise
                    executed.append(phase)
                    if phase.name == 'grasp':
                        local = driver.confirm_grasp()
                        carrying = True
                        adjusted = replace(current_plan, payload_local=local)
                        fresh = planning_view(replace(current_scene, cubes={k: v for k, v in current_scene.cubes.items() if k != plan.target_id}, captured_at=time.time()), params)   # verified unchanged before this phase
                        current_plan, planned_scene = planner.replan_held(fresh, driver.measured_joints(), adjusted, original)
                        current_scene = guard_view(planned_scene)
                        world = ExtractionCollisionWorld(backend, current_scene)
                        driver.preflight_remainder(current_plan, world, initial_joints=driver.measured_joints())
                        queue = list(current_plan.phases)
                        atomic_record(output / 'plan-after-grasp.json', plan_record(current_plan, backend))
                        log('post-grasp plan simulated and accepted')
                    if phase.name == 'release':
                        driver.release_confirmed()
                        carrying = False
                driver.mode = 'home'
                state['stage'] = 'FINISHED'
            publish_stage('FINISHED')
    finally:
        if owned:
            try:
                if json.loads(lease.read_text()).get('owner_nonce') == nonce:
                    lease.unlink()
            except (OSError, ValueError):
                pass


def run_cycle(args, config, calibration, backend, perception, planner, params, root_out, log, index):
    output = root_out / f'cycle-{index}'
    output.mkdir(parents=True)
    report = {'cycle': index, 'artifacts': str(output), 'success': False, 'motion_performed': False}
    try:
        scene, patches, info = perception.survey(mode=args.mode)
        inside = perception.inside_ids(scene)
        outside = [k for k, c in scene.cubes.items() if k not in inside and not c.evidence.get('unmeasured')]
        report.update(mode=args.mode, scene_info=info, cubes={k: c.public() for k, c in scene.cubes.items()}, inside=inside,
                      box={'center_xy_m': scene.box.base_from_box[:2, 3].tolist(), 'inner_size_m': (scene.box.inner_max_xy - scene.box.inner_min_xy).tolist(),
                           'yaw_rad': float(np.arctan2(scene.box.base_from_box[1, 0], scene.box.base_from_box[0, 0])), 'rim_height_m': float(scene.box.obstacle_top_m),
                           'flaps': scene.box.evidence.get('flaps')})
        frame = cv2.imread(str(Path(info['scans'][-1]) / 'frame.jpg'))
        if not (outside if args.mode == 'stow' else inside):
            report.update(success=True, empty=True)
            return report
        initial = perception_initial_joints(config, info, args)
        ready = choose_ready(backend, replace(scene, captured_at=time.time()))
        planner.command_sag = read_command_sag(config, initial, output)
        remaining = list(patches)
        if args.mode == 'extract' and isinstance(planner, NudgePlanner):
            ids = [k for k in inside if not scene.cubes[k].evidence.get('unmeasured')]
            probe = planning_view(replace(scene, placement=remaining[0], captured_at=time.time()),
                                  replace(params, flap_planning_margin_m=.005, planning_wall_clearance_m=.004))
            if ids and remaining and not planner.graspable(probe, ids, ready):
                if planner.nudges >= MAX_NUDGES:
                    raise Infeasible(f'no cube can be grasped where it lies and {MAX_NUDGES} pushes did not help')
                log('no cube in the box can be grasped where it lies: the arm will push one toward a place where it can be')
                try:
                    push, description = planner.plan_nudge(probe, ready, ids, log)
                except ValueError as exc:
                    raise Infeasible(f'no cube in the box can be grasped where it lies, and none can be pushed to a place where it can: {exc}') from exc
                trial = replace(scene, placement=remaining[0], captured_at=time.time())
                write_overlay(output / 'plan-overlay.jpg', frame, trial, None, calibration)
                atomic_record(output / 'plan.json', plan_record(push, backend))
                world = ExtractionCollisionWorld(backend, trial)
                try:
                    report['preflight'] = BoxArmDriver(config, backend, output).preflight_nudge(push, world, ready, initial_joints=initial)
                except Exception as exc:                          # noqa: BLE001 - the replay refused the push; nothing has moved
                    raise Infeasible(f'no cube in the box can be grasped where it lies; the simulated replay rejects the push that would move one: {exc}') from exc
                report.update(nudge=description, phases=[p.name for p in push.phases])
                log(f'push {description["distance_m"] * 1000:.0f} mm along {np.round(description["direction"], 2).tolist()} planned and simulated: '
                    f'{report["preflight"]["external_samples_checked"]} samples')
                if args.execute:
                    report['motion_performed'] = True
                    execute(args, config, calibration, backend, perception, planner, params, trial, push, ready, initial, output, log, base_scene=scene, patches=patches)
                    log('the push is done; the scene will be measured again')
                planner.nudges += 1
                report.update(nudged=True, success=True)
                return report
        for attempt in range(MAX_PLAN_ATTEMPTS):
            plan, trial, planned_params = plan_relaxing(planner, scene, remaining, ready, log, params)
            write_overlay(output / 'plan-overlay.jpg', frame, trial, plan, calibration)
            atomic_record(output / 'plan.json', plan_record(plan, backend))
            world = ExtractionCollisionWorld(backend, trial)
            driver = BoxArmDriver(config, backend, output)
            try:
                report['preflight'] = driver.preflight_full(plan, world, ready, initial_joints=initial, params=planned_params)
                params = planned_params
                break
            except (ValueError, RuntimeError) as exc:
                # The simulated replay (including the replan after the grasp) found a problem with this plan: take another place.
                remaining = [p for p in remaining if p is not trial.placement]
                log(f'the simulated replay rejected this plan ({str(exc)[:140]}); trying another place ({attempt + 1}/{MAX_PLAN_ATTEMPTS})')
                if attempt + 1 == MAX_PLAN_ATTEMPTS or not remaining:
                    raise Infeasible(f'the simulated replay rejects every plan: {exc}') from exc
        report.update(target=plan.target_id, placement_center_m=plan.placement_center_m.tolist(), phases=[p.name for p in plan.phases])
        log(f'preflight passed: {report["preflight"]["external_samples_checked"]} samples, target {plan.target_id}')
        if args.execute:
            # Before every phase (the first included) a fresh scan is compared with this measurement; a nudged box is followed.
            report['motion_performed'] = True
            execute(args, config, calibration, backend, perception, planner, params, trial, plan, ready, initial, output, log, base_scene=scene, patches=patches)
            report['placement'] = (perception.verify_stowed if args.mode == 'stow' else perception.verify_placement)(plan, len(inside), len(perception.measured(scene)))
            log(f'placement confirmed, error {report["placement"]["placement_error_m"] * 1000:.1f} mm')
        report['success'] = True
        return report
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        (output / 'result.json').write_text(json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


def perception_initial_joints(config, info, args):
    if args.execute:
        return ReadOnlySO101(config.robot['port'], config.robot['calibration']).read()['positions']
    last = json.loads((Path(info['scans'][-1]) / 'detections.json').read_text())
    if not last.get('robot_joint_deg'):
        raise RuntimeError('the scan carries no measured joints for the simulated start pose')
    return last['robot_joint_deg']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true', help='move the real arm (default: measure, plan, simulate only)')
    parser.add_argument('--mode', choices=('extract', 'stow'), default='extract', help='extract: cubes out of the box onto the table; stow: cubes from the table into the box')
    parser.add_argument('--confirm-attended', action='store_true', help='operator is present with STOP at hand')
    parser.add_argument('--max-cubes', type=int, default=0, help='stop after this many cubes (default 0: until the box is empty; without --execute one cycle is planned)')
    parser.add_argument('--wall-clearance-mm', type=float, default=3.)
    parser.add_argument('--position-sigma-mm', type=float, default=2.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-compat', default='single_cube_pick_place.py',
                        help='name recognised by the existing STOP button in this process command line')
    args = parser.parse_args()
    config = load_config(ROOT / 'config/default.json')
    calibration = json.loads((ROOT / config.raw['paths']['calibration']).read_text())
    if not calibration.get('validated') or float(calibration.get('reprojection_rmse_px', 1e9)) > float(config.safety['calibration_max_reprojection_px']):
        raise SystemExit('camera-to-robot calibration is not valid')
    if args.execute:
        require_attended(args)
    out = (args.output or HERE / 'logs' / stamp()).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_file = (out / 'log.txt').open('a')

    def log(message):
        line = f'{datetime.now().strftime("%H:%M:%S")} {message}'
        print(line, flush=True)
        log_file.write(line + '\n'); log_file.flush()
    params = SceneParameters(position_sigma_m=args.position_sigma_mm / 1000., vertical_sigma_m=args.position_sigma_mm / 1000.,
                             wall_clearance_m=args.wall_clearance_mm / 1000.,
                             rim_height_interval_m=tuple(config.raw['scene']['box_rim_height_interval_m']),
                             flap_top_m=float(config.raw['scene']['box_obstacle_height_upper_bound_m']))
    backend = So101ModelPlanner(config.robot['calibration'], table_z_m=float(calibration.get('table_z_m', 0.)),
                                offsets=config.robot.get('kinematic_offsets_deg'))
    if not np.allclose(backend.model.offsets, calibration.get('robot_offsets_deg', [0.] * 5), atol=1e-8):
        raise SystemExit('camera calibration and planner use different joint-zero offsets')
    perception = OverviewPerception(API, calibration, backend, params, log=log)
    planner = StowPlanner(backend) if args.mode == 'stow' else NudgePlanner(backend)
    log(f'mode: {"EXECUTE (real arm)" if args.execute else "dry run (no motor commands)"}; output {out}')
    extracted, cycles, retries = 0, 0, 0
    while extracted < (args.max_cubes or (1000 if args.execute else 1)):
        try:
            report = run_cycle(args, config, calibration, backend, perception, planner, params, out, log, cycles)
        except Infeasible as exc:
            if extracted:                                         # useful work was done; what is left cannot be done safely
                what = 'в коробке больше нет достижимого свободного места' if args.mode == 'stow' else 'оставшиеся кубики недостижимы'
                log(f'STOP: {extracted} cube(s) done, {what}. Details: {str(exc)[:300]}')
                break
            log(f'NO EXTRACTION POSSIBLE: {exc}')
            sys.exit(3)
        except (SceneError, SceneChanged) as exc:
            # Before the first working move (the arm only stands at its ready pose) a changed scene costs nothing: measure and
            # plan again. Later the arm is in or over the box and holds its pose.
            if (isinstance(exc, Restart) or getattr(exc, 'phase', None) in (None, 'open')) and retries < MAX_RESTARTS:
                retries += 1
                log(f'the scene changed before the first move ({exc}); measuring and planning again ({retries}/{MAX_RESTARTS})')
                continue
            log(f'STOPPED, scene not usable: {exc}')
            sys.exit(2)
        finally:
            cycles += 1
        if report.get('empty'):
            log('the table is clear: every cube is in the box' if args.mode == 'stow' else 'the box is empty')
            break
        if report.get('nudged'):
            if not args.execute:
                break                                             # a dry run only shows the push; the scene does not change
            retries = 0
            continue
        extracted += 1
        retries = 0
        planner.nudges = 0
    log(f'done: {extracted} cube(s) {"extracted" if args.execute else "planned and simulated"}')


if __name__ == '__main__':
    main()
