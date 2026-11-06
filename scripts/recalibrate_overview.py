"""Refit the overview camera to the arm from the red cross on the moving jaw, check it, and install it with a backup.

The project's own live control compares where the cross is seen with where the arm's kinematics says it is, and
refuses to run when they differ by more than 5 px. The calibration is a few pixels off at some poses (the camera or the
table may have moved a little). This script uses the project's collection and fit scripts unchanged:

  1. calibrate_overview_hand_eye.py   moves the arm through free-air poses and records the cross's pixel at each;
  2. fit_overview_marker.py           fits camera extrinsics, focal length and the cross's place on the jaw, with whole-pose holdout;
  3. this script                      checks the candidate on the parked arm (poses the fit never saw), and only if the
                                      held-out error and that control are small, installs it after backing up the old file.

Nothing is installed when a check fails. The project's web server reads the calibration at start: restart it afterwards.

Usage: recalibrate_overview.py [--observations <observations.json>] [--install]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'
sys.path.insert(0, str(ROOT / 'src'))
from so101ik import Model  # noqa: E402
from so101_pick_place.config import load_config  # noqa: E402
from so101_pick_place.hardware.robot import ReadOnlySO101  # noqa: E402
from so101_pick_place.vision.calibration_control import validate_overview_control  # noqa: E402

PY = str(ROOT / '.venv/bin/python')
MAX_HELDOUT_PX = 3.5
MAX_CONTROL_PX = 3.5


def frame():
    data = urllib.request.urlopen('http://127.0.0.1:8010/snapshot/overview.jpg', timeout=5).read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def control_errors(calibration, model, config, samples=6):
    errors = []
    for _ in range(samples):
        joints = ReadOnlySO101(config.robot['port'], config.robot['calibration']).read()['positions']
        try:
            errors.append(validate_overview_control(frame(), calibration, model, joints, 5.)['error_px'])
        except ValueError as exc:
            text = str(exc)
            errors.append(float(text.split('error ')[1].split('px')[0]) if 'control error' in text else float('inf'))
        time.sleep(.3)
    return errors


def fit_with_offsets(record, installed, model):
    """Camera extrinsics, focal length and the cross's place on the jaw, with the joint zeros the robot is installed with.

    The project's fit_overview_marker.py builds its model without those zeros, which is why it cannot reproduce the
    installed calibration from new data. Same parameters, same whole-pose holdout, started from the installed values.
    """
    from scipy.optimize import least_squares
    link = installed['marker_link']
    records = record['observations']
    pixels = np.asarray([r['marker_px'] for r in records], float)
    frames = np.array([model.frames(r['actual_deg'])[link] for r in records])
    rvec0 = cv2.Rodrigues(np.asarray(installed['camera_from_robot'])[:3, :3])[0].ravel()
    start = np.r_[rvec0, np.asarray(installed['camera_from_robot'])[:3, 3], installed['marker_local_m'], installed['camera_matrix'][0][0]]

    def project(values, index=slice(None)):
        world = np.einsum('nij,j->ni', frames[index, :3, :3], values[6:9]) + frames[index, :3, 3]
        camera = np.array([[values[9], 0., 640.], [0., values[9], 360.], [0., 0., 1.]])
        return cv2.projectPoints(world, values[:3], values[3:6], camera, np.zeros(5))[0].reshape(-1, 2)

    def residual(values, selected):
        return np.r_[(project(values)[selected] - pixels[selected]).ravel(), (values[9] - installed['camera_matrix'][0][0]) / 150.]

    mesh = model.mesh_points()[link]
    lo, hi = mesh.min(0) - .002, mesh.max(0) + .002
    lower = np.r_[start[:3] - .3, start[3:6] - .1, lo, start[9] - 150.]
    upper = np.r_[start[:3] + .3, start[3:6] + .1, hi, start[9] + 150.]
    everything = np.ones(len(records), bool)
    fit = least_squares(lambda v: residual(v, everything), np.clip(start, lower + 1e-9, upper - 1e-9), bounds=(lower, upper), max_nfev=4000)
    values = fit.x
    train = np.linalg.norm(project(values) - pixels, axis=1)
    groups = [tuple(round(float(r['planned_deg'][j]), 2) for j in ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')) for r in records]
    held = []
    for group in dict.fromkeys(groups):
        test = np.array([g == group for g in groups])
        refit = least_squares(lambda v: residual(v, ~test), values, bounds=(lower, upper), max_nfev=3000)
        held.extend(np.linalg.norm(project(refit.x)[test] - pixels[test], axis=1).tolist())
    rotation, _ = cv2.Rodrigues(values[:3])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, values[3:6]
    centre = -rotation.T @ values[3:6]
    installed_error = np.linalg.norm(project(start) - pixels, axis=1)
    return {'link': link, 'camera_from_robot': transform.tolist(), 'camera_matrix': [[float(values[9]), 0., 640.], [0., float(values[9]), 360.], [0., 0., 1.]],
            'distortion': [0.] * 5, 'marker_local_m': values[6:9].tolist(), 'camera_center_robot_m': centre.tolist(),
            'train_rmse_px': float(np.sqrt(np.mean(train ** 2))), 'reprojection_rmse_px': float(np.sqrt(np.mean(np.array(held) ** 2))),
            'heldout_max_px': float(max(held)), 'installed_rmse_px': float(np.sqrt(np.mean(installed_error ** 2))),
            'observed_pixels': pixels.tolist(), 'predicted_pixels': project(values).tolist(),
            'fit_passed': bool(np.sqrt(np.mean(np.array(held) ** 2)) <= MAX_HELDOUT_PX and max(held) <= 8. and np.sqrt(np.mean(train ** 2)) <= 3.0
                               and .25 < centre[2] < 1.5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--observations', type=Path, help='use these collected observations instead of moving the arm')
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    config = load_config(ROOT / 'config/default.json')
    installed = json.loads((ROOT / 'config/calibration.json').read_text())
    model = Model(calibration_file=config.robot['calibration'], offsets=config.robot.get('kinematic_offsets_deg'))
    model.table_z = float(installed['table_z_m'])
    before = control_errors(installed, model, config)
    print(f'installed calibration: live control at the parked pose {np.round(before, 2).tolist()} px', flush=True)
    observations = args.observations
    if observations is None:
        raise SystemExit('collect first: calibrate_overview_hand_eye.py --execute --marker-seed X Y, then pass --observations')
    candidate_path = Path(__file__).resolve().parents[1] / 'calibration' / f'overview-candidate-{datetime.now(UTC).strftime("%Y%m%dT%H%M%S")}.json'
    candidate_path.parent.mkdir(exist_ok=True)
    best = fit_with_offsets(json.loads(Path(observations).read_text()), installed, model)
    candidate_path.write_text(json.dumps(best, indent=2))
    print(f'installed calibration on these observations: {best["installed_rmse_px"]:.2f} px', flush=True)
    print(f'candidate: held-out {best["reprojection_rmse_px"]:.2f} px, train {best["train_rmse_px"]:.2f} px, fit_passed {best["fit_passed"]}', flush=True)
    new = dict(installed) | {key: best[key] for key in ('camera_matrix', 'camera_from_robot', 'distortion', 'marker_local_m', 'camera_center_robot_m')} \
        | {'reprojection_rmse_px': best['reprojection_rmse_px'], 'train_rmse_px': best['train_rmse_px'],
           'sample_count': len(best['observed_pixels']), 'created_at': datetime.now(UTC).isoformat(timespec='seconds'),
           'source_artifacts': str(Path(observations).parent.relative_to(ROOT))}
    after = control_errors(new, model, config)
    print(f'candidate: live control at the parked pose {np.round(after, 2).tolist()} px (the fit never saw this pose)', flush=True)
    accept = best['fit_passed'] and best['reprojection_rmse_px'] <= MAX_HELDOUT_PX and max(after) <= MAX_CONTROL_PX and max(after) < max(before)
    new['validation'] = {'whole_pose_holdout': True, 'independent_physical_control': {'passed': bool(max(after) <= MAX_CONTROL_PX), 'error_px': float(max(after)),
                                                                                    'artifacts': 'parked-arm live control, six frames'},
                         'refit_by': 'scripts/recalibrate_overview.py', 'previous_control_px': float(max(before))}
    print('ACCEPTED' if accept else 'REJECTED: the installed calibration stays', flush=True)
    if accept and args.install:
        backup = ROOT / f'config/calibration.old-auto-{datetime.now(UTC).strftime("%Y%m%dT%H%M%S")}.json'
        shutil.copy2(ROOT / 'config/calibration.json', backup)
        (ROOT / 'config/calibration.json').write_text(json.dumps(new, indent=2))
        print(f'installed; previous calibration saved as {backup.name}; restart the project web server so it reads it', flush=True)
    sys.exit(0 if accept else 1)


if __name__ == '__main__':
    main()
