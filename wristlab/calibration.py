"""Fit measured calibration photographs; never installs an unvalidated candidate."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .vision import color_masks,projective_center


def refine_face_edges(contour, quad):
    points = np.asarray(contour,float).reshape(-1,2)
    q = np.asarray(quad,float).reshape(4,2)
    lines = []
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        edge = b-a;length = np.linalg.norm(edge);direction = edge/length
        along = (points-a)@direction
        normal = np.array([-direction[1],direction[0]])
        distance = np.abs((points-a)@normal)
        selected = points[(along>.2*length)&(along<.8*length)&(distance<max(3.,.06*length))]
        if len(selected)<8:
            raise ValueError('not enough straight top-edge pixels')
        fit = cv2.fitLine(selected.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
        lines.append((fit[:2],fit[2:]))
    corners = []
    for i in range(4):
        da,pa = lines[i-1];db,pb = lines[i]
        matrix = np.column_stack((da,-db))
        if abs(np.linalg.det(matrix))<.1:
            raise ValueError('nearly parallel top edges')
        alpha,_ = np.linalg.solve(matrix,pb-pa)
        corners.append(pa+alpha*da)
    return np.asarray(corners)


def top_observation(image, color):
    contours,_ = cv2.findContours(color_masks(image)[color],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('calibration colour is absent')
    contour = max(contours,key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(contour)
    if cv2.contourArea(contour)<300 or min(x,y)<3 or x+w>=image.shape[1]-3 or y+h>=image.shape[0]-3:
        raise ValueError('calibration face is clipped or too small')
    hull = cv2.convexHull(contour)
    quad = cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)
    if len(quad)!=4:
        raise ValueError('colour mask is not one complete planar quadrilateral')
    refined = refine_face_edges(contour,quad)
    return projective_center(refined),refined


def load_center_observations(paths, model, measured_side):
    observations,rejected,targets = [],[],[]
    for dataset,path in enumerate(paths):
        path = Path(path)
        source = json.loads(path.read_text())
        if not source.get('completed') or not np.allclose(source.get('robot_offsets_deg',[]),model.offsets):
            raise ValueError('incomplete calibration or different joint zeros')
        if abs(source['gripper_encoder_span_deg']-model.gripper_span_deg)>1e-6:
            raise ValueError('calibration uses a different gripper encoder span')
        image = cv2.imread(str(path.parent/'reference-overview.jpg'))
        x1,y1,x2,y2 = source['reference_cube']['box_xyxy']
        scores = {color:int(np.count_nonzero(mask[y1:y2,x1:x2])) for color,mask in color_masks(image).items()}
        color = max(scores,key=scores.get)
        if scores[color]<300:
            raise ValueError('reference colour cannot be identified independently')
        vertices = np.asarray(source['world_vertices_m'])
        top = vertices[4:].mean(0)
        top[2] = float(source.get('fixture_height_m',0.))+measured_side
        targets.append(top)
        for item in source['observations']:
            image_path = path.parent/f"pose-{item['index']:02d}-wrist.jpg"
            image = cv2.imread(str(image_path))
            try:
                pixel,quad = top_observation(image,color)
                observations.append({'dataset':dataset,'group':f"{dataset}:{item['group']}",'q':item['actual_deg'],
                    'pixel':pixel,'quad':quad,'source_image':str(image_path)})
            except ValueError as exc:
                rejected.append({'source_image':str(image_path),'reason':str(exc)})
    return observations,np.asarray(targets),rejected


def fit_center_hand_eye(observations, targets, model, seed, parent):
    if len(observations)<20:
        raise ValueError('insufficient complete measured calibration observations')
    frames = np.asarray([model.frames(o['q'])[parent] for o in observations])
    pixels = np.asarray([o['pixel'] for o in observations])
    labels = np.asarray([o['dataset'] for o in observations])
    groups = list(dict.fromkeys(o['group'] for o in observations))
    withheld = set(groups[2::4])
    test = np.asarray([o['group'] in withheld for o in observations]);train = ~test
    if len(groups)<8 or test.sum()<3:
        raise ValueError('independent whole-pose holdout is insufficient')
    rotations = np.asarray([Rotation.from_matrix(frames[0,:3,:3].T@f[:3,:3]).as_rotvec() for f in frames[1:]])
    motion = np.linalg.svd(rotations,compute_uv=False)
    if motion[1]<np.deg2rad(8.) or np.linalg.norm(np.ptp(frames[:,:3,3],axis=0))<.025:
        raise ValueError('camera mounting is not observable from recorded motion')
    mount = np.asarray(seed['parent_from_camera'])
    k = np.asarray(seed['camera_matrix'])
    initial = np.r_[Rotation.from_matrix(mount[:3,:3]).as_rotvec(),mount[:3,3],np.log([k[0,0],k[1,1]]),k[0,2],k[1,2],seed['distortion'][:4],np.zeros(targets.size)]
    lower = np.r_[[-2*np.pi]*3,[-.2]*3,np.log([250.,250.]),256.,144.,-.8,-.8,-.05,-.05,np.tile([-.015,-.015,-.005],len(targets))]
    upper = np.r_[[2*np.pi]*3,[.2]*3,np.log([2000.,2000.]),1024.,576.,.8,.8,.05,.05,np.tile([.015,.015,.005],len(targets))]
    initial = np.clip(initial,lower+1e-6,upper-1e-6)
    def project(p):
        rotation = Rotation.from_rotvec(p[:3]).as_matrix()
        cam_rotation = frames[:,:3,:3]@rotation
        cam_center = np.einsum('nij,j->ni',frames[:,:3,:3],p[3:6])+frames[:,:3,3]
        world = targets[labels]+p[14:].reshape(-1,3)[labels]
        optical = np.einsum('nji,nj->ni',cam_rotation,world-cam_center)
        x,y = (optical[:,:2]/optical[:,2,None]).T
        radius = x*x+y*y
        radial = 1+p[10]*radius+p[11]*radius*radius
        dx = 2*p[12]*x*y+p[13]*(radius+2*x*x)
        dy = p[12]*(radius+2*y*y)+2*p[13]*x*y
        return np.column_stack((np.exp(p[6])*(x*radial+dx)+p[8],np.exp(p[7])*(y*radial+dy)+p[9])),optical[:,2]
    def residual(p,selected):
        projected,depth = project(p)
        offset = p[14:].reshape(-1,3)/[.004,.004,.002]
        return np.r_[(projected[selected]-pixels[selected]).ravel(),np.minimum(0.,depth[selected]-.025)*1000.,offset.ravel(),p[12:14]/.02]
    fit = least_squares(lambda p:residual(p,train),initial,bounds=(lower,upper),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=500)
    uv,depth = project(fit.x);errors = np.linalg.norm(uv-pixels,axis=1)
    validation_rmse = float(np.sqrt(np.mean(errors[test]**2)))
    validation_max = float(errors[test].max())
    jac = fit.jac[:2*int(train.sum())]
    norms = np.maximum(np.linalg.norm(jac,axis=0),1e-12)
    singular = np.linalg.svd(jac/norms,compute_uv=False)
    condition = float(singular[0]/max(singular[-1],1e-15))
    reasons = []
    if validation_rmse>2.5 or validation_max>5.:
        reasons.append('withheld-pose projection exceeds 2.5/5px')
    if errors.max()>5.:
        reasons.append('one or more measured views are inconsistent beyond 5px')
    if np.linalg.matrix_rank(jac/norms,tol=1e-5)<len(fit.x):
        reasons.append('data Jacobian has unobservable parameter directions')
    if np.any(np.minimum(fit.x-lower,upper-fit.x)<1e-4):
        reasons.append('fit reaches a physical parameter boundary')
    if condition>1e5:
        reasons.append('hand-eye and intrinsics are not independently observable')
    if depth.min()<.025:
        reasons.append('reference point behind or too close to camera')
    deltas = fit.x[14:].reshape(-1,3)
    if np.max(np.linalg.norm(deltas[:,:2],axis=1))>.006:
        reasons.append('recorded metric target correction exceeds 6mm')
    p=fit.x;transform=np.eye(4);transform[:3,:3]=Rotation.from_rotvec(p[:3]).as_matrix();transform[:3,3]=p[3:6]
    return {'validated':not reasons,'failure_reasons':reasons,'parent_link':parent,'parent_from_camera':transform.tolist(),
      'camera_matrix':[[float(np.exp(p[6])),0.,float(p[8])],[0.,float(np.exp(p[7])),float(p[9])],[0.,0.,1.]],
      'distortion':[*p[10:14].tolist(),0.],'robot_offsets_deg':list(model.offsets),'gripper_encoder_span_deg':model.gripper_span_deg,
      'image_size':[1280,720],'validation_rmse_px':validation_rmse,'validation_max_px':validation_max,
      'all_observation_max_px':float(errors.max()),'per_observation_error_px':errors.tolist(),
      'normalised_jacobian_condition':condition,'target_adjustments_m':deltas.tolist(),
      'independent_control':{'passed':False},'calibration_target_not_used_for_runtime_grasp':True,
      'method':'projective coloured-top centres, recorded measured FK, multiple stationary calibration targets and whole-pose holdout'}
