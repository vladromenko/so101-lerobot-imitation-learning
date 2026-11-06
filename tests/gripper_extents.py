import json,sys
from pathlib import Path
import numpy as np
ROOT=Path.home()/"Projects/so101_autonomous_pick_place"
from so101_pick_place.kinematics import So101ModelPlanner
config=json.loads((ROOT/"config/default.json").read_text()); cal=json.loads((ROOT/"config/calibration.json").read_text())
b=So101ModelPlanner(config["robot"]["calibration"],table_z_m=0.,offsets=config["robot"]["kinematic_offsets_deg"])
q={"shoulder_pan":0.,"shoulder_lift":0.,"elbow_flex":0.,"wrist_flex":0.,"wrist_roll":0.}
for g in (28.0,):
    fr=b.model.frames(dict(q,gripper=g)); tip=fr["gripper_frame_link"]
    for name in ("gripper_link","moving_jaw_so101_v1_link","wrist_link"):
        pts=b._transform_points(np.linalg.inv(tip)@fr[name], b.meshes[name])
        print(name,"gripper",g,"min mm",np.round(pts.min(0)*1000,1),"max mm",np.round(pts.max(0)*1000,1))
