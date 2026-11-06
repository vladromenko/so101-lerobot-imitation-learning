"""Derive a *synthetic* scan from a real one: one cube moved to a chosen spot in the box.
Used only for offline pipeline tests; the output is labelled synthetic."""
import json, sys
import numpy as np
scan = json.load(open(sys.argv[1])); dx, dy = float(sys.argv[3]), float(sys.argv[4])
cx, cy = None, None
for d in scan['detections']:
    if d['label'] == 'box_opening':
        cx, cy = d['drop_geometry']['xy_m']
cubes = [d for d in scan['detections'] if d['label'] == 'cube' and d.get('inside_box')]
keep = []
for d in scan['detections']:
    if d['label'] == 'cube' and d.get('inside_box'):
        if d is cubes[0]:
            continue                                  # drop the yellow cube
        g = d['geometry']; side = g['height_m']
        g['center_volume_m'] = [cx + dx, cy + dy, side / 2]; g['center_top_m'] = [cx + dx, cy + dy, side]
        g['axes_robot_xy'] = [[1., 0.], [0., 1.]]
        d['synthetic'] = True
    keep.append(d)
scan['detections'] = keep
scan['synthetic'] = True
json.dump(scan, open(sys.argv[2], 'w'))
