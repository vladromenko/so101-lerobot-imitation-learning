import json,sys,cv2,numpy as np
from pathlib import Path
HERE=Path(".").resolve(); sys.path.insert(0,str(HERE))
ROOT=Path.home()/"Projects/so101_autonomous_pick_place"
from so101_pick_place.kinematics import So101ModelPlanner
from box_extract.scene_builder import *
scan=json.load(open("tmp/synth_center.json")); fr=cv2.imread("tmp/frame1.jpg")
cfg=json.loads((ROOT/"config/default.json").read_text()); cal=json.loads((ROOT/"config/calibration.json").read_text())
b=So101ModelPlanner(cfg["robot"]["calibration"],table_z_m=0.,offsets=cfg["robot"]["kinematic_offsets_deg"])
p=SceneParameters(wall_clearance_m=.003)
sc,pa,info=build_scene([json.loads(json.dumps(scan)) for _ in range(3)],[fr]*3,cal,b,p,revision="x",ids=IdentityTracker())
for name,view in (("guard",guard_view(sc)),("plan",planning_view(sc,p))):
    print(name,"clearance",view.box.clearance_m,"inflate",view.box.flap_inflate_m)
    for lo,hi,label in view.box.solid_parts():
        if "flap" in label or "wall" in label: print("  ",label,np.round(lo,3).tolist(),np.round(hi,3).tolist())
print("faces",sc.box.flap_faces)
