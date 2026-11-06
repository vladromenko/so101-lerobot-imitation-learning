"""Forward and inverse kinematics of the SO-101 written out step by step, with the robot's real numbers.

Everything is computed from the URDF the controller itself uses and checked against the compiled library
(`so101ik`), so what the tutorial shows is what the robot computes.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

URDF = Path.home() / 'Projects/so101_autonomous_pick_place/vendor/SO101/so101_new_calib.urdf'
ARM_ORDER = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')


def parse_urdf(path=URDF):
    text = Path(path).read_text()
    joints = {}
    for match in re.finditer(r'<joint name="(\w+)" type="(\w+)">(.*?)</joint>', text, re.S):
        body = match.group(3)
        origin = re.search(r'<origin xyz="([^"]+)" rpy="([^"]+)"', body)
        axis = re.search(r'<axis xyz="([^"]+)"', body)
        joints[match.group(1)] = {
            'type': match.group(2),
            'parent': re.search(r'<parent link="(\w+)"', body).group(1),
            'child': re.search(r'<child link="(\w+)"', body).group(1),
            'xyz': [float(v) for v in origin.group(1).split()],
            'rpy': [float(v) for v in origin.group(2).split()],
            'axis': [float(v) for v in axis.group(1).split()] if axis else [0., 0., 1.],
        }
    return joints


def rot_x(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
def rot_y(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
def rot_z(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def fixed_transform(xyz, rpy):
    """URDF origin: translate by xyz, rotate by R = Rz(yaw) Ry(pitch) Rx(roll)."""
    T = np.eye(4)
    T[:3, :3] = rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])
    T[:3, 3] = xyz
    return T


def joint_transform(joint, angle_rad):
    T = np.eye(4)
    axis = np.asarray(joint['axis'], float)
    assert np.allclose(axis, [0, 0, 1]), 'every SO-101 joint turns about its own z axis'
    T[:3, :3] = rot_z(angle_rad)
    return T


def fk_steps(joints_deg, offsets_deg, urdf=None):
    """Each link's frame as the product of (fixed origin) x (rotation about z by the joint angle)."""
    urdf = urdf or parse_urdf()
    steps, T = [], np.eye(4)
    for name in ARM_ORDER:
        joint = urdf[name]
        angle = math.radians(joints_deg[name] + offsets_deg[ARM_ORDER.index(name)])
        fixed, turn = fixed_transform(joint['xyz'], joint['rpy']), joint_transform(joint, angle)
        T = T @ fixed @ turn
        steps.append({'joint': name, 'parent': joint['parent'], 'child': joint['child'], 'xyz': joint['xyz'], 'rpy': joint['rpy'],
                      'angle_deg': joints_deg[name] + offsets_deg[ARM_ORDER.index(name)], 'fixed': fixed.tolist(), 'turn': turn.tolist(),
                      'cumulative': T.copy().tolist()})
    tool = urdf['gripper_frame_joint']
    T = T @ fixed_transform(tool['xyz'], tool['rpy'])
    steps.append({'joint': 'gripper_frame_joint', 'parent': tool['parent'], 'child': tool['child'], 'xyz': tool['xyz'], 'rpy': tool['rpy'],
                  'angle_deg': None, 'fixed': fixed_transform(tool['xyz'], tool['rpy']).tolist(), 'turn': np.eye(4).tolist(), 'cumulative': T.tolist()})
    return steps


def planar_geometry(urdf=None):
    """Lengths that matter once the pan has turned the arm into a vertical plane."""
    urdf = urdf or parse_urdf()
    upper = np.asarray(urdf['elbow_flex']['xyz'][:2], float)       # shoulder_lift -> elbow_flex, in the upper arm's frame
    lower = np.asarray(urdf['wrist_flex']['xyz'][:2], float)       # elbow_flex -> wrist_flex, in the lower arm's frame
    return {'upper_arm_m': float(np.linalg.norm(upper)), 'upper_offset_deg': math.degrees(math.atan2(upper[1], upper[0])),
            'lower_arm_m': float(np.linalg.norm(lower)), 'lower_offset_deg': math.degrees(math.atan2(lower[1], lower[0]))}


# ---------------------------------------------------------------------------------------------------------------
# inverse kinematics, in the three ways the system uses it

def tip_and_pitch(frames):
    frame = np.asarray(frames['gripper_frame_link'])
    z_axis = frame[:3, 2]
    return frame[:3, 3], math.degrees(math.asin(max(-1., min(1., z_axis[2]))))


def ik_scan(model, target, pitches=range(-90, -9, 5)):
    """The library's closed-form IK for one tip target at a fixed pitch; the cost is the gravity load it would hold."""
    rows = []
    for pitch in pitches:
        try:
            pose = model.ik(*target, pitch=float(pitch))
            rows.append({'pitch': pitch, 'ok': True, 'deg': {k: round(float(v), 2) for k, v in pose.deg.items()}, 'cost': round(float(pose.cost), 4),
                         'tip': [round(float(v), 4) for v in pose.tip]})
        except Exception as exc:                                    # Unreachable: no allowed pose reaches the point
            rows.append({'pitch': pitch, 'ok': False, 'reason': str(exc)[:80]})
    best = None
    try:
        pose = model.ik(*target)
        best = {'deg': {k: round(float(v), 2) for k, v in pose.deg.items()}, 'pitch': round(float(pose.pitch), 2), 'cost': round(float(pose.cost), 4)}
    except Exception as exc:
        best = {'error': str(exc)[:100]}
    return {'target': list(target), 'rows': rows, 'best': best}


def elbow_triangle(model, joints):
    """The shoulder-elbow-wrist triangle of the arm's vertical plane, from the model's own frames."""
    frames = model.frames(joints)
    S = np.asarray(frames['upper_arm_link'])[:3, 3]
    E = np.asarray(frames['lower_arm_link'])[:3, 3]
    W = np.asarray(frames['wrist_link'])[:3, 3]
    L1, L2, d = float(np.linalg.norm(E - S)), float(np.linalg.norm(W - E)), float(np.linalg.norm(W - S))
    gamma = math.degrees(math.acos((L1 ** 2 + L2 ** 2 - d ** 2) / (2 * L1 * L2)))
    alpha = math.degrees(math.acos((L1 ** 2 + d ** 2 - L2 ** 2) / (2 * L1 * d)))
    return {'S': S.tolist(), 'E': E.tolist(), 'W': W.tolist(), 'L1': L1, 'L2': L2, 'd': d, 'gamma_deg': gamma, 'alpha_deg': alpha}


def fk_tip(urdf, offsets_deg, q_deg):
    T = np.eye(4)
    for name, angle in zip(ARM_ORDER, q_deg):
        joint = urdf[name]
        T = T @ fixed_transform(joint['xyz'], joint['rpy']) @ joint_transform(joint, math.radians(angle + offsets_deg[ARM_ORDER.index(name)]))
    tool = urdf['gripper_frame_joint']
    T = T @ fixed_transform(tool['xyz'], tool['rpy'])
    return T


def newton_iterations(urdf, offsets_deg, target, pitch_deg, seed_deg, free=(0, 1, 2, 3), iterations=12, damping=1e-3):
    """Damped Gauss-Newton on the forward kinematics: what the planner's `least_squares` does to polish a grasp pose.

    Unknowns: the four joints that place the tip and set the pitch (roll is left alone). Residual: tip position error in
    millimetres and the pitch error in degrees (weighted so that 1 degree weighs like 1 mm).
    """
    q = np.asarray(seed_deg, float).copy()

    def residual(q):
        T = fk_tip(urdf, offsets_deg, q)
        pitch = math.degrees(math.asin(max(-1., min(1., T[2, 2]))))
        return np.r_[(T[:3, 3] - np.asarray(target)) * 1000., pitch - pitch_deg], T

    rows = []
    for k in range(iterations + 1):
        r, _ = residual(q)
        rows.append({'k': k, 'q': [round(float(v), 4) for v in q], 'error_mm': round(float(np.linalg.norm(r[:3])), 4), 'pitch_error_deg': round(float(r[3]), 4),
                     'residual': [round(float(v), 4) for v in r]})
        if k == iterations or np.linalg.norm(r) < 1e-6:
            break
        J = np.zeros((4, len(free)))
        for j, index in enumerate(free):
            step = np.zeros(5)
            step[index] = 1e-4
            J[:, j] = (residual(q + step)[0] - residual(q - step)[0]) / 2e-4
        delta = -np.linalg.solve(J.T @ J + damping * np.eye(len(free)), J.T @ r)
        rows[-1]['jacobian'] = J.round(3).tolist()
        rows[-1]['step'] = [round(float(v), 4) for v in delta]
        for j, index in enumerate(free):
            q[index] += delta[j]
    return rows
