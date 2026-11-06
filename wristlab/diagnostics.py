"""Hardware limits, sampled model margins and unapproved escape proposals."""
import json
import time
import numpy as np


def minimum_clearance(model, joints, ceiling=.02, resolution=.0001):
    pose = model.pose(joints)
    if model.collides(pose,gap=0.) is not None:
        return 0.
    lo,hi = 0.,float(ceiling)
    while hi-lo > resolution:
        middle = (lo+hi)/2
        if model.collides(pose,gap=middle) is None:
            lo = middle
        else:
            hi = middle
    return lo


def audit_pose(robot, state):
    m,q = robot.backend.model,state['positions']
    pose = m.pose(q)
    hard = m.problems(pose,margin=0.)
    exact = m.collides(pose,gap=0.)
    hardware = [name+' outside encoder-calibrated range' for name,(lo,hi) in zip(
        ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'),m.ranges) if not lo<=q[name]<=hi]
    if not 0.<=q['gripper']<=100.:
        hardware.append('gripper outside encoder-calibrated range')
    separation = minimum_clearance(m,q)
    return {'timestamp':state['timestamp'],'joints_native':q,'q_rad':robot.from_native(q).tolist(),
            'hard_limit_problems':hard,'hardware_limit_problems':hardware,
            'hardware_limit_status':'VIOLATED' if hardware else 'WITHIN_LIMITS',
            'actual_mesh_collision':None,'mesh_intersection_status':'NOT_ESTABLISHED',
            'zero_gap_sample_contact':exact,'collision_method':'native sampled surfaces; zero gap is not an exact triangle intersection test',
            'planning_margin_problems':m.problems(pose),'planning_clearance_collision':m.collides(pose),
            'software_margin_status':'VIOLATED' if m.problems(pose) or m.collides(pose) else 'SATISFIED',
            'required_joint_margin_rad':np.deg2rad(m.margin),'required_clearance_m':float(m.gap),
            'minimum_model_clearance_m':separation,'physical_clearance_measured':False,
            'minimum_sample_pair':m.collides(pose,gap=separation+.0001),
            'hard_valid':not hard and exact is None,'planning_valid':not m.problems(pose) and m.collides(pose) is None,
            'tip_m':list(pose.tip),'kinematic_offsets_native':list(m.offsets)}


def propose_separating_escape(robot, state, check_cancel=lambda:None):
    """Only shoulder/elbow change; wrist flex, roll and jaw remain measured.

    This evaluates a recovery corridor that monotonically leaves an existing
    soft-margin violation. It does not authorize a controller bypass, and it
    never certifies accessories or an unmeasured cable.
    """
    model,start = robot.backend.model,state['positions']
    audit = audit_pose(robot,state)
    if not audit['hard_valid']:
        return {'software_path_found':False,'physical_authorized':False,'reason':'hard limit or true mesh intersection at start','audit':audit}
    base_clearance = audit['minimum_model_clearance_m']
    choices = []
    for amount in (.105,.14,.175,.21,.26):
        for ratio in (.8,1.,1.2):
            check_cancel()
            delta = np.array([0.,amount,-amount*ratio,0.,0.])
            goal = robot.to_native(robot.from_native(start)+delta,start['gripper'])
            end = model.pose(goal)
            if not model.problems(end) and model.collides(end) is None:
                samples = list(robot.native_edge_samples(start,goal))
                clearances = []
                safe = True
                for q in samples:
                    pose = model.pose(q)
                    clearance = minimum_clearance(model,q,ceiling=.012,resolution=.0002)
                    clearances.append(clearance)
                    if model.problems(pose,margin=0.) or model.collides(pose,gap=0.) or clearance < base_clearance-.0002:
                        safe = False
                monotonic = all(b >= a-.0002 for a,b in zip(clearances,clearances[1:]))
                if safe and monotonic:
                    choices.append((float(np.linalg.norm(delta)),samples,clearances))
    if not choices:
        return {'software_path_found':False,'physical_authorized':False,'reason':'no monotonically separating shoulder/elbow corridor found','audit':audit}
    _,samples,clearances = min(choices,key=lambda c:c[0])
    return {'software_path_found':True,'physical_authorized':False,'audit':audit,
            'path_native':samples,'clearances_m':clearances,'goal_native':samples[-1],
            'fixed_axes':['wrist_flex','wrist_roll','gripper'],
            'remaining_checks':['measured camera/mount on complete swept path','cable swept volume, beyond current visual inspection',
                                'gravity-compensated commands and tracking envelope','attended controlled noncontact trial'],
            'automatic_unfold_used':False}
