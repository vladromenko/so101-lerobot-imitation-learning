"""Hand-eye fit of the wrist camera from the coloured top face of a stationary cube.

SAM 3 segments only the coloured top face of printed cubes in wrist frames, not the whole cube, and the
project's silhouette fit assumes whole-cube silhouettes. The top face is a planar quadrilateral whose four
corners the overview camera measured; this script passes it to the project's own `fit_silhouette_hand_eye`
as a cube 1 mm high, so the optimiser, the withheld-pose validation and the physical-admissibility gates are
the project's own. Output has the same layout as scripts/fit_wrist_camera.py.

Usage: fit_wrist_top_face.py <observations.json> --out <candidate.json>
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'
sys.path.insert(0, str(ROOT / 'src'))
from so101ik import JOINTS, Model  # noqa: E402
from so101_pick_place.vision.wrist_silhouette_calibration import fit_silhouette_hand_eye  # noqa: E402

THIN_M = .001


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.observations.read_text())
    config = json.loads((ROOT / 'config/default.json').read_text())
    model = Model(calibration_file=config['robot']['calibration'], offsets=config['robot']['kinematic_offsets_deg'])
    if source.get('completed') is not True or source.get('observation_kind') != 'cube_silhouette':
        raise SystemExit('a completed stationary-cube observation set is required')
    if not np.allclose(source['robot_offsets_deg'], model.offsets, atol=1e-8):
        raise SystemExit('the robot zero calibration changed since image acquisition')
    observations = source['observations']
    if np.ptp([o['actual_deg']['gripper'] for o in observations]) < 15.:
        raise SystemExit('different measured gripper openings are required to identify the camera parent link')
    directory = args.observations.resolve().parent
    masks = [cv2.imread(str(directory / o['cube_mask_png']), cv2.IMREAD_GRAYSCALE) for o in observations]
    top = np.asarray(source['world_vertices_m'], float)[4:]
    world = np.vstack((top - [0., 0., THIN_M], top))
    candidates = []
    for parent in ('gripper_frame_link', 'moving_jaw_so101_v1_link'):
        try:
            frames = [model.frames(o['actual_deg'])[parent] for o in observations]
            result = fit_silhouette_hand_eye(frames, masks, world, source['image_size'], groups=[o['group'] for o in observations])
            result.update(method='coloured top face of a stationary cube, measured FK, joint intrinsic/hand-eye/target-pose fit',
                          parent_link=parent, robot_offsets_deg=list(model.offsets), gripper_encoder_span_deg=model.gripper_span_deg,
                          source_observations=str(args.observations.resolve()), independent_live_control_required=True,
                          fit_kind='top_face_silhouette')
            candidates.append(result)
            print(parent, result['validated'], round(result['validation_rmse_px'], 2), round(result['validation_max_px'], 2),
                  round(result['all_observation_rmse_px'], 2), result['failure_reasons'], flush=True)
        except ValueError as exc:
            candidates.append({'parent_link': parent, 'validated': False, 'error': str(exc)})
            print(parent, 'error', exc, flush=True)
    valid = [c for c in candidates if c.get('validated')]
    best = valid[0] if len(valid) == 1 else None
    reasons = []
    if not valid:
        reasons.append('no camera-parent hypothesis passed global and withheld-pose validation')
    elif len(valid) > 1:
        reasons.append('ambiguous camera parent: both mounting hypotheses passed')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'candidate': best, 'parent_candidates': candidates, 'installed': False, 'failure_reasons': reasons}, indent=2))
    if best is None:
        raise SystemExit('; '.join(reasons))


if __name__ == '__main__':
    main()
