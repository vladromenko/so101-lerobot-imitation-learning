"""Run one of the project's own wrist-camera scripts, adapted to this scene (no change to the project's files).

The project's calibration scripts observe the stationary cube from a fixed list of camera poses
tuned for a cube about 14 cm from the robot base, and take the box opening from the server. This
wrapper executes the script's own source with two changes made in memory:

* the observation offsets are shifted toward the base by as much as the cube lies beyond comfortable
  reach, and the tool-pitch fallbacks are widened, so a cube anywhere on the table gets enough
  reachable views (every view still goes through the script's own IK, path and collision preflight);
* the server's scan replies are completed with the rim-edge box opening (tools/sitecustomize.py)
  when the server finds none, and cubes inside that opening are marked inside the box.

Usage: run_project_calibrator.py <project script> [its arguments]
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
os.environ['SO101_OPENING_FALLBACK'] = '1'
import importlib.util  # noqa: E402

# The shim file is named sitecustomize.py so it also works through PYTHONPATH; a distribution may
# already have imported its own `sitecustomize`, so load this one by path.
_spec = importlib.util.spec_from_file_location('so101_scan_shim', HERE / 'tools/sitecustomize.py')
importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

API = 'http://127.0.0.1:8010'
COMFORT_RADIUS_M = .20          # tool-tip distance from the base where the observation poses are easy to reach
MAX_SHIFT_M = .12


def scene_summary(require_cube=True):
    """Where the cube the project's stable_scene will pick (nearest outside cube) and the box opening are."""
    for _ in range(8):        # also primes the opening tracker, so the project's own scans see a steady opening
        scan = json.load(urllib.request.urlopen(urllib.request.Request(API + '/api/scan-hybrid', method='POST', data=b''), timeout=120))
    cubes = [d for d in scan['detections'] if d['label'] == 'cube' and not d.get('inside_box') and d.get('geometry') and d.get('score', 0.) >= .8]
    if not cubes:
        if require_cube:
            raise SystemExit('no measured cube outside the box to observe')
        return {'cube_xy': None, 'opening_xy': None}
    cube = min(cubes, key=lambda d: sum(v * v for v in d['geometry']['center_volume_m'][:2]))
    openings = [d for d in scan['detections'] if d['label'] == 'box_opening' and d.get('drop_geometry')]
    return {'cube_xy': cube['geometry']['center_volume_m'][:2], 'opening_xy': openings[0]['drop_geometry']['xy_m'] if openings else None}


def adapted_source(source, summary):
    cube = [float(v) for v in summary['cube_xy']]
    radius = (cube[0] ** 2 + cube[1] ** 2) ** .5
    shift = -min(max(radius - COMFORT_RADIUS_M, 0.), MAX_SHIFT_M)
    views_block = re.search(r'^VIEWS=\(\n(.*?)^\)\n', source, flags=re.S | re.M)
    control_line = re.search(r'^CONTROL_VIEWS=(\(.*\))$', source, flags=re.M)
    if not views_block or not control_line:
        raise SystemExit('the project script no longer has the expected VIEWS/CONTROL_VIEWS tables; nothing was changed')
    views = eval('(' + views_block.group(1) + ')')
    control = eval(control_line.group(1))
    # Keep the camera on the side of the cube away from the box: the project refuses poses near the box volume.
    side = 0.
    if summary['opening_xy'] is not None:
        radial = [cube[0] / radius, cube[1] / radius]
        tangent = [-radial[1], radial[0]]
        toward_box = (summary['opening_xy'][0] - cube[0]) * tangent[0] + (summary['opening_xy'][1] - cube[1]) * tangent[1]
        side = -1. if toward_box > 0 else 1.
    moved = lambda table: tuple((dr + shift, (side * (abs(dt) + .01) if side else dt), *rest) for dr, dt, *rest in table)
    source = source.replace(views_block.group(0), 'VIEWS=' + repr(moved(views)) + '\n')
    source = source.replace(control_line.group(0), 'CONTROL_VIEWS=' + repr(moved(control)))
    source = source.replace('fallback_pitches=(-85.,-75.,-65.)', 'fallback_pitches=(-85.,-75.,-65.,-55.,-45.)')
    print(f'[run_project_calibrator] reference cube {radius * 100:.1f} cm from the base: observation offsets shifted {shift * 100:.1f} cm radially', file=sys.stderr)
    return source


def main():
    script = Path(sys.argv[1]).resolve()
    if script.parent != (Path.home() / 'Projects/so101_autonomous_pick_place/scripts'):
        raise SystemExit('only the project\'s own scripts can be run through this wrapper')
    args = sys.argv[2:]
    source = script.read_text()
    summary = scene_summary(require_cube='VIEWS=' in source and '--install' not in args)          # also primes the opening tracker, so the first scans the script takes see a steady opening
    if 'VIEWS=' in source and '--install' not in args:
        source = adapted_source(source, summary)
    namespace = {'__name__': '__main__', '__file__': str(script), '__builtins__': __builtins__}
    sys.argv = [str(script), *args]
    sys.path.insert(0, str(script.parent))
    exec(compile(source, str(script), 'exec'), namespace)


if __name__ == '__main__':
    main()
