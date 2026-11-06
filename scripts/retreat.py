"""Bring the arm back along an already validated plan path after a safe stop (the arm holds its pose).

The program stops with the arm holding its pose when a scene check fails. The path of the phase that brought
the arm there was planned and simulated collision-free against the measured scene, so the way back is that
path in reverse, slowly, with every tick required to stay within a few degrees of it. The arm ends at the
path's first pose and keeps holding; it does not park.

Usage: retreat.py --plan logs/<run>/cycle-0/plan.json [--phase approach] [--stop-compat NAME]
"""
import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'
sys.path.insert(0, str(ROOT / 'src'))
from so101_pick_place.config import load_config  # noqa: E402
from so101_pick_place.hardware.robot import ReadOnlySO101  # noqa: E402
from so101_pick_place.hardware.task_arm import TaskArm  # noqa: E402
from so101_pick_place.kinematics import So101ModelPlanner  # noqa: E402
from so101_pick_place.kinematics.so101_backend import ARM_JOINTS  # noqa: E402

MAX_DEVIATION_DEG = 3.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', default='approach', help='comma-separated phases in driving order, e.g. approach,enter')
    parser.add_argument('--keep-path-gripper', action='store_true', help='keep the jaw goal the plan recorded (holding a cube) instead of the measured angle')
    parser.add_argument('--open-gripper', action='store_true', help='first open the jaws in place to the path\'s own opening (cube clamped)')
    parser.add_argument('--stop-compat', default='single_cube_pick_place.py')
    args = parser.parse_args()
    config = load_config(ROOT / 'config/default.json')
    robot = config.robot
    backend = So101ModelPlanner(robot['calibration'], table_z_m=0., offsets=robot.get('kinematic_offsets_deg'))
    plan = json.loads(args.plan.read_text())
    path = [q for name in args.phase.split(',') for q in next(p['joints'] for p in plan['phases'] if p['name'] == name)]
    now = ReadOnlySO101(robot['port'], robot['calibration']).read()['positions']
    deviation = [max(abs(q[j] - now[j]) for j in ARM_JOINTS) for q in path]
    nearest = int(np.argmin(deviation))
    if deviation[nearest] > MAX_DEVIATION_DEG:
        raise SystemExit(f'the arm is {deviation[nearest]:.1f} deg from the plan path; refusing to retrace it')
    open_angle = float(path[nearest]['gripper'])
    way_back = [dict(path[i], gripper=open_angle if args.open_gripper else (path[i]['gripper'] if args.keep_path_gripper else now['gripper'])) for i in range(nearest, -1, -1)]
    print(f'retreating along {args.phase}: waypoint {nearest} -> 0 ({len(way_back)} poses), arm {deviation[nearest]:.2f} deg from the path', flush=True)
    poses = [backend.model.pose(q) for q in way_back]
    joints = np.array([[p.deg[j] for j in ARM_JOINTS] for p in poses])
    joint_steps = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
    tip_steps = np.linalg.norm(np.diff(np.array([p.tip for p in poses]), axis=0), axis=1)
    ratios = joint_steps / np.maximum(tip_steps, 1e-7)
    duration = max(.5, 1.875 * float(tip_steps.sum()) / min(.008, 5. / max(1e-6, float(ratios.max()))))
    on_path = np.array([[q[j] for j in ARM_JOINTS] for q in path])

    def guard(tick):
        for kind in ('pos', 'goal'):
            sample = np.array([tick[kind][j] for j in ARM_JOINTS])
            if np.min(np.max(np.abs(on_path - sample), axis=1)) > MAX_DEVIATION_DEG:
                raise RuntimeError(f'retreat {kind} left the validated path by more than {MAX_DEVIATION_DEG} deg')

    lease = ROOT / 'runtime/single_cube_executor.json'
    if lease.exists():
        raise SystemExit('another executor owns the servo bus')
    descriptor = os.open(lease, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    nonce = uuid.uuid4().hex
    with os.fdopen(descriptor, 'w') as handle:
        json.dump({'pid': os.getpid(), 'script': Path(__file__).name, 'owner_nonce': nonce}, handle)
    try:
        arm = TaskArm(port=robot['port'], calibration_file=robot['calibration'], offsets=robot.get('kinematic_offsets_deg'),
                      on_tick=guard, hold_on_success=True, out=HERE / 'logs/retreat', name='retreat')
        with arm:
            if args.open_gripper:
                actual, _, _ = arm.read()
                arm.move_joints(dict(actual, gripper=open_angle), speed=6., check=True, step='retreat_open')
                print(f'jaws opened to {open_angle:.1f}', flush=True)
            arm.follow(poses, duration=duration, settle=.4, within=.7, lead=False, step='retreat')
        print('retreat finished; the arm holds its pose', flush=True)
    finally:
        try:
            if json.loads(lease.read_text()).get('owner_nonce') == nonce:
                lease.unlink()
        except (OSError, ValueError):
            pass


if __name__ == '__main__':
    main()
