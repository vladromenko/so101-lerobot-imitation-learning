"""Web interface for the box program: run it (stow all cubes into the box / take all cubes out) and learn how it computes.

Served from the Jetson; open http://homebrain.local:8030 on the Mac. Two tabs:
  * Работа: two buttons that start the whole program, live camera, log, STOP;
  * Как это считается: forward and inverse kinematics with the robot's real numbers, a 3D view in its coordinate
    frames, the camera model, the collision test and path search (RRT-Connect), including the last real plan.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parents[2]
PROJECT = Path.home() / 'Projects/so101_autonomous_pick_place'
PY = str(PROJECT / '.venv/bin/python')
API = 'http://127.0.0.1:8010'
STATIC = Path(__file__).resolve().parent / 'static'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / 'src'))

from box_extract.ui import kin_explain as K  # noqa: E402

app = FastAPI(title='SO-101: коробка')


# ---------------------------------------------------------------------------------------------------- run control
class Runner:
    """One child process at a time: the whole program in one of its modes."""

    def __init__(self):
        self.proc, self.mode, self.dry, self.started, self.finished, self.log_path = None, None, None, None, None, None
        self.lock = threading.Lock()

    def command(self, mode, dry):
        wrapper = str(HERE / 'scripts/run_project_calibrator.py')
        if mode == 'stow':
            return [PY, '-u', str(HERE / 'scripts/extract_from_box.py'), '--mode', 'stow', '--stop-compat', 'single_cube_pick_place.py', *([] if dry else ['--execute', '--confirm-attended'])]
        if mode == 'extract':
            return [PY, '-u', str(HERE / 'scripts/extract_from_box.py'), '--stop-compat', 'single_cube_pick_place.py', *([] if dry else ['--execute', '--confirm-attended'])]
        raise HTTPException(400, 'unknown mode')

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, mode, dry):
        with self.lock:
            if self.running():
                raise HTTPException(409, 'программа уже работает')
            if (PROJECT / 'runtime/single_cube_executor.json').exists():
                raise HTTPException(409, 'шину сервоприводов держит другой процесс (runtime/single_cube_executor.json)')
            (HERE / 'logs/ui').mkdir(parents=True, exist_ok=True)
            self.log_path = HERE / 'logs/ui' / f'{datetime.now().strftime("%Y%m%dT%H%M%S")}-{mode}{"-dry" if dry else ""}.log'
            log = self.log_path.open('w')
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            self.proc = subprocess.Popen(self.command(mode, dry), cwd=HERE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
            self.mode, self.dry, self.started, self.finished = mode, dry, time.time(), None

    def stop(self):
        if self.running():
            os.killpg(self.proc.pid, signal.SIGINT)          # the program holds the arm's pose, like the project's STOP button
        try:                                                    # and the project's own STOP, for the executors it recognises by name
            urllib.request.urlopen(urllib.request.Request(API + '/api/stop', method='POST', data=b''), timeout=3).read()
        except Exception:
            pass

    def status(self):
        running = self.running()
        if self.proc is not None and not running and self.finished is None:
            self.finished = time.time()
        tail = []
        if self.log_path and self.log_path.exists():
            lines = [l for l in self.log_path.read_text(errors='replace').splitlines() if 'opening-fallback' not in l]
            tail = lines[-160:]
        project = {}
        try:
            project = json.load(urllib.request.urlopen(API + '/api/status', timeout=2))
        except Exception:
            pass
        return {'running': running, 'mode': self.mode, 'dry': self.dry, 'started': self.started, 'finished': self.finished,
                'exit_code': None if self.proc is None or running else self.proc.returncode, 'log': tail,
                'phase': current_phase(tail), 'project_state': project.get('state'), 'project_detail': project.get('detail')}


def current_phase(lines):
    for line in reversed(lines):
        for key in ('placement confirmed', 'the box is empty', 'done:', 'STOPPED', 'NO EXTRACTION POSSIBLE'):
            if key in line:
                return line.split(' ', 1)[-1][:120]
        if ' phase ' in line:
            return line.split(' ', 1)[-1].split(':')[0]
        if '--- ' in line:
            return line.split('--- ', 1)[-1][:100]
    return None


runner = Runner()


@app.post('/api/run')
def run(payload: dict):
    mode, dry, attended = payload.get('mode'), bool(payload.get('dry')), bool(payload.get('attended'))
    if not dry and not attended:
        raise HTTPException(400, 'для работы с рукой нужно подтвердить, что вы рядом и можете нажать STOP')
    runner.start(mode, dry)
    return {'started': True}


@app.post('/api/stop')
def stop():
    runner.stop()
    return {'stopped': True}


@app.get('/api/status')
def status():
    return runner.status()


# ---------------------------------------------------------------------------------------------------- the scene
_scene_lock = threading.Lock()


@app.get('/api/scene')
def scene():
    if runner.running():
        raise HTTPException(409, 'пока работает программа, сцену не измеряю (камера занята распознаванием)')
    with _scene_lock:
        from box_extract.server_compat import complete_scan_record
        record = None
        for _ in range(3):
            record = complete_scan_record(json.load(urllib.request.urlopen(urllib.request.Request(API + '/api/scan-hybrid', method='POST', data=b''), timeout=120)))
        cubes = [{'px': d['center_px'], 'xy_m': (d.get('position_robot_m') or [None, None])[:2], 'inside': bool(d.get('inside_box')), 'score': d['score'],
                  'polygon': d.get('silhouette_polygons_px', [])} for d in record['detections'] if d['label'] == 'cube']
        openings = [d for d in record['detections'] if d['label'] == 'box_opening']
        return {'cubes': cubes, 'inside': sum(c['inside'] for c in cubes), 'outside': sum(not c['inside'] for c in cubes),
                'opening': openings[0]['silhouette_polygons_px'][0] if openings else None, 'time': time.time()}


# ---------------------------------------------------------------------------------------------------- the tutorial
_cache = {}


def model():
    if 'model' not in _cache:
        from so101ik import Model
        from so101_pick_place.config import load_config
        config = load_config(PROJECT / 'config/default.json')
        _cache['config'] = config
        _cache['model'] = Model(calibration_file=config.robot['calibration'], offsets=config.robot.get('kinematic_offsets_deg'))
        _cache['urdf'] = K.parse_urdf()
        _cache['calibration'] = json.loads((PROJECT / 'config/calibration.json').read_text())
    return _cache['model']


def joints_from(payload):
    base = {'shoulder_pan': 0., 'shoulder_lift': -30., 'elbow_flex': 40., 'wrist_flex': 60., 'wrist_roll': 0., 'gripper': 30.}
    return base | {k: float(v) for k, v in (payload or {}).items() if k in base}


@app.get('/api/robot')
def robot():
    m = model()
    if 'meshes' not in _cache:
        meshes = json.load(urllib.request.urlopen(API + '/api/model-geometry', timeout=20))['meshes']
        _cache['meshes'] = {k: {'vertices': np.round(np.asarray(v['vertices']), 4).tolist(), 'triangles': v['triangles']} for k, v in meshes.items()}
    urdf = _cache['urdf']
    return {'joints': {k: urdf[k] for k in (*K.ARM_ORDER, 'gripper_frame_joint')}, 'ranges_deg': np.asarray(m.ranges).tolist(), 'offsets_deg': list(m.offsets),
            'planar': K.planar_geometry(urdf), 'meshes': _cache['meshes'], 'camera': {'matrix': _cache['calibration']['camera_matrix'],
            'camera_from_robot': _cache['calibration']['camera_from_robot']}}


@app.post('/api/fk')
def fk(payload: dict):
    m = model()
    q = joints_from(payload)
    steps = K.fk_steps(q, list(m.offsets), _cache['urdf'])
    frames = {k: np.asarray(v).tolist() for k, v in m.frames(q).items()}
    err = max(float(np.abs(np.asarray(s['cumulative']) - np.asarray(frames[name])).max())
              for s, name in zip(steps, ('shoulder_link', 'upper_arm_link', 'lower_arm_link', 'wrist_link', 'gripper_link', 'gripper_frame_link')))
    tip, pitch = K.tip_and_pitch(frames)
    return {'joints': q, 'steps': steps, 'frames': frames, 'max_difference_to_library_m': err, 'tip': tip.tolist(), 'pitch_deg': pitch,
            'triangle': K.elbow_triangle(m, q)}


@app.post('/api/ik')
def ik(payload: dict):
    m = model()
    target = [float(payload.get(k, d)) for k, d in (('x', .2), ('y', -.1), ('z', .08))]
    out = K.ik_scan(m, target)
    best = out['best']
    if 'deg' in best:
        q = [best['deg'][j] for j in K.ARM_ORDER]
        out['triangle'] = K.elbow_triangle(m, best['deg'])
        zero = [0., 0., 0., 0., 0.]
        # interior angle at the elbow when the elbow joint's URDF angle is 0, to turn gamma into the joint angle
        frames0 = m.frames({**best['deg'], 'elbow_flex': -float(m.offsets[2])})
        S, E, W = (np.asarray(frames0[n])[:3, 3] for n in ('upper_arm_link', 'lower_arm_link', 'wrist_link'))
        L1, L2, d = np.linalg.norm(E - S), np.linalg.norm(W - E), np.linalg.norm(W - S)
        out['gamma0_deg'] = math.degrees(math.acos((L1 ** 2 + L2 ** 2 - d ** 2) / (2 * L1 * L2)))
        out['elbow_offset_deg'] = float(m.offsets[2])
        seed = np.asarray(q) + np.array([8., -6., 10., -9., 0.])
        out['newton'] = K.newton_iterations(_cache['urdf'], list(m.offsets), target, best['pitch'], seed)
        out['seed_deg'] = seed.tolist()
    return out


@app.post('/api/pixel')
def pixel(payload: dict):
    """Pixel -> point on a horizontal plane, written out: ray through the pinhole, intersection with the plane."""
    model()
    cal = _cache['calibration']
    K_ = np.asarray(cal['camera_matrix'], float)
    T = np.asarray(cal['camera_from_robot'], float)
    u, v, z = float(payload.get('u', 640)), float(payload.get('v', 360)), float(payload.get('z', 0.))
    ray_cam = np.array([(u - K_[0, 2]) / K_[0, 0], (v - K_[1, 2]) / K_[1, 1], 1.])
    R, t = T[:3, :3], T[:3, 3]
    centre = -R.T @ t
    direction = R.T @ ray_cam
    lam = (z - centre[2]) / direction[2]
    point = centre + lam * direction
    return {'u': u, 'v': v, 'z': z, 'camera_matrix': K_.tolist(), 'camera_from_robot': T.tolist(), 'ray_camera': ray_cam.tolist(),
            'camera_centre_robot': centre.tolist(), 'ray_robot': direction.tolist(), 'lambda': float(lam), 'point': point.tolist(),
            'reprojection_rmse_px': cal.get('reprojection_rmse_px')}


def urdf_visuals():
    """Per link: the STL pieces that draw it (origin in the link frame, colour), as in the URDF the LeLab viewer loads."""
    import re
    text = K.URDF.read_text()
    colors = {m.group(1): [float(v) for v in m.group(2).split()] for m in re.finditer(r'<material name="(\w+)">\s*<color rgba="([^"]+)"', text)}
    links = {}
    for match in re.finditer(r'<link name="(\w+)">(.*?)</link>', text, re.S):
        pieces = []
        for visual in re.findall(r'<visual>(.*?)</visual>', match.group(2), re.S):
            origin = re.search(r'<origin xyz="([^"]+)" rpy="([^"]+)"', visual)
            mesh = re.search(r'filename="([^"]+)"', visual)
            material = re.search(r'<material name="(\w+)"', visual)
            if mesh:
                pieces.append({'mesh': Path(mesh.group(1)).name, 'xyz': [float(v) for v in origin.group(1).split()] if origin else [0, 0, 0],
                               'rpy': [float(v) for v in origin.group(2).split()] if origin else [0, 0, 0], 'rgba': colors.get(material.group(1) if material else '', [.8, .8, .8, 1.])})
        links[match.group(1)] = pieces
    return links


@app.get('/api/urdf')
def urdf():
    model()
    return {'links': urdf_visuals(), 'joints': {k: _cache['urdf'][k] for k in (*K.ARM_ORDER, 'gripper_frame_joint', 'gripper')}}


@app.get('/mesh/{name}')
def mesh(name: str):
    path = PROJECT / 'vendor/SO101/assets' / Path(name).name
    if not path.exists():
        raise HTTPException(404, 'no such mesh')
    return FileResponse(path, media_type='model/stl', headers={'Cache-Control': 'max-age=86400'})


@app.get('/api/live')
def live():
    """The real arm's joints as the running program publishes them (read-only), with the kinematics worked out for them."""
    t = json.load(urllib.request.urlopen(API + '/api/telemetry', timeout=3))
    sample = t.get('sample') or {}
    pos = sample.get('pos')
    out = {'time': sample.get('time'), 'age_s': time.time() - float(sample.get('time') or 0), 'pos': pos, 'source': sample.get('source'), 'math': t.get('math')}
    if pos:
        q = joints_from(pos)
        m = model()
        frames = {k: np.asarray(v).tolist() for k, v in m.frames(q).items()}
        tip, pitch = K.tip_and_pitch(frames)
        out |= {'frames': frames, 'tip': tip.tolist(), 'pitch_deg': pitch}
    return out


@app.get('/api/plan')
def plan():
    """The newest real plan, from any run: every phase's waypoints (joints and the tip), the box and the cubes it was planned for."""
    candidates = sorted((HERE / 'logs').glob('**/cycle-*/plan.json'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise HTTPException(404, 'ещё не было ни одного плана')
    path = candidates[-1]
    data = json.loads(path.read_text())
    phases = data['phases']
    after = path.parent / 'plan-after-grasp.json'
    if after.exists() and after.stat().st_mtime >= path.stat().st_mtime - 1:          # the phases after the grasp as they were really driven
        replanned = {p['name']: p for p in json.loads(after.read_text())['phases']}
        phases = [replanned.get(p['name'], p) if p['name'] in replanned else p for p in phases]
    result = {}
    try:
        result = json.loads((path.parent / 'result.json').read_text())
    except Exception:
        pass
    stride = lambda items: items[::2] if len(items) > 120 else items
    return {'path': str(path.relative_to(HERE)), 'when': path.stat().st_mtime, 'mode': result.get('mode'),
            'plan': {'target_id': data['target_id'], 'placement_center_m': data['placement_center_m'], 'searches': data['searches'],
                     'phases': [{'name': p['name'], 'tips': stride(p['path_tip_m']), 'joints': stride(p['joints']), 'carrying': p['carrying']} for p in phases]},
            'cubes': {k: {'center_m': v['center_m'], 'side_m': v['side_m'], 'vertices_m': v['vertices_m'], 'unmeasured': bool(v.get('evidence', {}).get('unmeasured'))} for k, v in result.get('cubes', {}).items()},
            'box': result.get('box'), 'inside': result.get('inside')}


@app.post('/api/clearance')
def clearance(payload: dict):
    """How far each link of the arm is from the cubes, the box walls and the table, for given joints and the newest scene."""
    m = model()
    if 'backend' not in _cache:
        from so101_pick_place.kinematics import So101ModelPlanner
        _cache['backend'] = So101ModelPlanner(_cache['config'].robot['calibration'], table_z_m=0., offsets=_cache['config'].robot.get('kinematic_offsets_deg'))
    backend = _cache['backend']
    info = plan()
    q = joints_from(payload)
    frames = m.frames(q)
    cubes = []
    for key, cube in info['cubes'].items():
        v = np.asarray(cube['vertices_m'], float)
        axes = np.array([v[1] - v[0], v[3] - v[0], v[4] - v[0]])
        half = np.linalg.norm(axes, axis=1) / 2.
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        cubes.append((key, v.mean(0), axes, half))
    box = info.get('box') or {}
    out = []
    for link, mesh in backend.meshes.items():
        if link not in frames:
            continue
        T = np.asarray(frames[link])
        points = (np.asarray(mesh)[::6] @ T[:3, :3].T) + T[:3, 3]
        row = {'link': link, 'table_mm': float(points[:, 2].min() * 1000.), 'cube': None, 'box_mm': None}
        best = None
        for key, centre, axes, half in cubes:
            local = (points - centre) @ axes.T
            d = np.linalg.norm(np.maximum(np.abs(local) - half, 0.), axis=1).min() * 1000.
            if best is None or d < best[0]:
                best = (d, key)
        if best:
            row['cube'] = {'mm': float(best[0]), 'which': best[1]}
        if box.get('center_xy_m'):
            cx, cy = box['center_xy_m']; sx, sy = box['inner_size_m']; yaw = box.get('yaw_rad', 0.); h = box.get('rim_height_m', .073); w = .01
            c, sn = math.cos(yaw), math.sin(yaw)
            rel = points - np.array([cx, cy, 0.])
            local = np.c_[rel[:, 0] * c + rel[:, 1] * sn, -rel[:, 0] * sn + rel[:, 1] * c, rel[:, 2]]
            slabs = [((sx / 2, -sy / 2 - w, 0.), (sx / 2 + w, sy / 2 + w, h)), ((-sx / 2 - w, -sy / 2 - w, 0.), (-sx / 2, sy / 2 + w, h)),
                     ((-sx / 2, sy / 2, 0.), (sx / 2, sy / 2 + w, h)), ((-sx / 2, -sy / 2 - w, 0.), (sx / 2, -sy / 2, h))]
            d = min(np.linalg.norm(np.maximum(np.maximum(np.asarray(lo) - local, local - np.asarray(hi)), 0.), axis=1).min() for lo, hi in slabs) * 1000.
            row['box_mm'] = float(d)
        out.append(row)
    return {'links': out, 'plan': info['path']}


@app.get('/api/sag')
def sag():
    """Gravity compensation at the last plan's grasp pose: the registers are commanded `sag x torque` degrees past the goal."""
    m = model()
    plans = sorted((HERE / 'logs').glob('**/cycle-*/plan.json'), key=lambda p: p.stat().st_mtime)
    if not plans:
        raise HTTPException(404, 'ещё не было ни одного плана')
    grasp = next(p for p in json.loads(plans[-1].read_text())['phases'] if p['name'] == 'grasp')['joints'][0]
    sag_value = 3.7                                           # deg per N*m, the C driver's setting (read from a simulated arm by the program)
    pose = m.pose(grasp)
    torque = [float(t) for t in pose.torque]                  # N*m on shoulder_pan ... wrist_roll
    command = dict(grasp)
    for joint, load in zip(K.ARM_ORDER[1:], torque[1:]):
        command[joint] = float(command[joint]) + sag_value * load
    f0, f1 = np.asarray(m.frames(grasp)['gripper_frame_link']), np.asarray(m.frames(command)['gripper_frame_link'])
    local = (f1[:3, 3] - f0[:3, 3]) @ f0[:3, :3]
    rotation = math.degrees(math.acos(max(-1., min(1., (np.trace(f0[:3, :3].T @ f1[:3, :3]) - 1) / 2))))
    return {'sag_deg_per_nm': sag_value, 'grasp': {k: round(float(v), 2) for k, v in grasp.items()}, 'torque_nm': [round(t, 4) for t in torque],
            'command_minus_goal_deg': {j: round(command[j] - grasp[j], 3) for j in K.ARM_ORDER}, 'shift_in_gripper_frame_mm': (local * 1000).round(2).tolist(),
            'rotation_deg': round(rotation, 2), 'plan': str(plans[-1].relative_to(HERE))}


@app.get('/snapshot.jpg')
def snapshot():
    try:
        return Response(urllib.request.urlopen(API + '/snapshot/overview.jpg', timeout=4).read(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html', headers={'Cache-Control': 'no-store'})


app.mount('/static', StaticFiles(directory=STATIC), name='static')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8030)), log_level='warning')
