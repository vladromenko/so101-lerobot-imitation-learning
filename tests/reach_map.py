"""Where in the box can the gripper still grasp a cube? For a grid of cube positions (the measured cube of a saved scan moved
there, same yaw), try every grasp candidate's vertical entry from above the rim and report why entries fail.
usage: reach_map.py scan.json frame.jpg [yaw_deg ...]"""
import json, sys, time, collections
from dataclasses import replace
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'scripts'))
ROOT = Path.home() / 'Projects/so101_autonomous_pick_place'; sys.path.insert(0, str(ROOT / 'scripts'))
from so101_pick_place.config import load_config
from so101_pick_place.kinematics import So101ModelPlanner
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld
from box_extract.scene_builder import *
from box_extract.planning import TunedExtractionPlanner
from box_extract.opening_fallback import estimate_opening
from box_extract.driver import read_command_sag
import extract_from_box as cli

scan = json.loads(Path(sys.argv[1]).read_text()); frame = cv2.imread(sys.argv[2])
if not any(d['label'] == 'box_opening' for d in scan['detections']):
    opening, _ = estimate_opening(frame, scan); scan['detections'].append(opening)
config = load_config(ROOT / 'config/default.json'); cal = json.loads((ROOT / 'config/calibration.json').read_text())
backend = So101ModelPlanner(config.robot['calibration'], table_z_m=0., offsets=config.robot['kinematic_offsets_deg'])
params = SceneParameters(wall_clearance_m=.003, flap_planning_margin_m=.024, planning_wall_clearance_m=.004)
scene, patches, info = build_scene([json.loads(json.dumps(scan)) for _ in range(3)], [frame] * 3, cal, backend, params, revision='offline', ids=IdentityTracker())
print('inside', info['inside_box'], 'box', np.round(scene.box.base_from_box[:3, 3], 3), 'inner', scene.box.inner_min_xy, scene.box.inner_max_xy)
ready = cli.choose_ready(backend, replace(scene, captured_at=time.time()))
planner = TunedExtractionPlanner(backend); planner.command_sag = read_command_sag(config, scan['robot_joint_deg'], Path('/tmp/reach'))
key = info['inside_box'][0]; cube0 = scene.cubes[key]
yaws = [float(a) for a in sys.argv[3:]] or [0.]
inv = np.linalg.inv(scene.box.base_from_box)
def moved(center_box_xy, yaw_deg):
    c = scene.box.base_from_box @ np.r_[center_box_xy, cube0.center_m[2] - 0., 1.]       # box frame -> base (z kept: cube on the floor)
    c = np.r_[c[:2], cube0.center_m[2]]
    yaw = np.deg2rad(yaw_deg) + np.arctan2(scene.box.base_from_box[1, 0], scene.box.base_from_box[0, 0])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.]])
    h = cube0.side_m / 2.; unit = np.array([[x, y, z] for z in (-h, h) for x, y in ((-h, -h), (h, -h), (h, h), (-h, h))])
    return replace(cube0, center_m=c, rotation=R, vertices_m=unit @ R.T + c)
lo, hi = scene.box.inner_min_xy, scene.box.inner_max_xy
for yaw in yaws:
    print(f'--- yaw {yaw} deg relative to the box')
    for fx in np.arange(.05, .96, .1):
        for fy in np.arange(.15, .86, .1):
            xy = lo + (hi - lo) * np.array([fx, fy]); cube = moved(xy, yaw)
            view = planning_view(replace(scene, cubes={**scene.cubes, key: cube}), params); world = ExtractionCollisionWorld(backend, view)
            policy = {'target': key, 'jaw_contact': True}; n = ok = 0; why = collections.Counter(); t = time.time()
            try:
                for grasp, closed in planner.grasp_candidates(view, view.cubes[key], ready):
                    n += 1
                    fr = planner.model.frames(grasp)['gripper_frame_link']; local = (view.cubes[key].vertices_m - fr[:3, 3]) @ fr[:3, :3]
                    rim = (view.box.base_from_box @ np.r_[0., 0., view.box.obstacle_top_m, 1.])[2]
                    rise = max(.025, rim + view.box.clearance_m + planner.rim_margin_m + cube.side_m / 2 - cube.center_m[2])
                    try:
                        up = planner.vertical_center(grasp, rise, local.mean(0), world, **policy)
                        bad = None
                        for q in up:
                            r = world.reason(planner._command_pose(q), **policy)
                            if r: bad = 'command ' + r; break
                        bad = bad or planner._tracking_robust(up, world, **policy)
                        if bad: why[bad.split(': ', 1)[-1][:60]] += 1
                        else: ok += 1
                    except ValueError as e: why[str(e)[:50]] += 1
                    if n >= 12: break
            except ValueError as e: why['no candidates: ' + str(e)[:50]] += 1
            print(f'box-frac ({fx:.2f},{fy:.2f}) xy {np.round(cube.center_m[:2], 3).tolist()}: {ok}/{n} entries ok  {dict(why.most_common(3))}  {time.time() - t:.0f}s', flush=True)
