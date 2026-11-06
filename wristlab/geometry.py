"""Calibrated overview rays and explicit upright-cube top-face fitting."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from .types import ObjectState


class Geometry:
    def __init__(self, native_root):
        self.path = Path(native_root) / 'config/calibration.json'
        self.calibration = json.loads(self.path.read_text())
        if not self.calibration.get('validated') or self.calibration.get('reprojection_rmse_px', 99) > 5.:
            raise ValueError('overview calibration is not validated')
        self.fingerprint = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.intrinsic = np.asarray(self.calibration['camera_matrix'], float)
        self.distortion = np.asarray(self.calibration['distortion'], float)
        self.camera_from_base = np.asarray(self.calibration['camera_from_robot'], float)
        self.table_z = float(self.calibration['table_z_m'])
        self.wrist = None
        reference = Path(native_root)/'config/grasp_tcp_calibration.json'
        self.known_cube = json.loads(reference.read_text()) if reference.exists() else None
        if self.known_cube is not None and (not self.known_cube.get('validated') or not .015 < self.known_cube.get('cube_size_m',0.) < .065):
            self.known_cube = None

    def camera_pose_in_base(self, q, camera_id, robot=None):
        if camera_id == 'overview':
            return np.linalg.inv(self.camera_from_base)
        if self.wrist is None or robot is None:
            raise ValueError('wrist-to-link calibration has not passed independent control')
        return robot.link_pose(q, self.wrist['parent_link']) @ np.asarray(self.wrist['parent_from_camera'])

    def pixel_ray_in_base(self, pixel, frame, robot=None):
        if frame.camera_id != 'overview':
            projection = self.projection_for_frame(frame,robot)
            xy = cv2.undistortPoints(np.asarray(pixel,float).reshape(1,1,2),np.asarray(projection['camera_matrix']),np.asarray(projection['distortion'])).ravel()
            pose = np.linalg.inv(np.asarray(projection['camera_from_robot']))
            direction = pose[:3,:3]@np.r_[xy,1.]
            return pose[:3,3],direction/np.linalg.norm(direction)
        xy = cv2.undistortPoints(np.asarray(pixel, float).reshape(1, 1, 2), self.intrinsic, self.distortion).reshape(2)
        pose = np.linalg.inv(self.camera_from_base)
        direction = pose[:3, :3] @ np.r_[xy, 1.]
        return pose[:3, 3], direction / np.linalg.norm(direction)

    @staticmethod
    def intersect_ray_with_plane(ray, plane):
        origin, direction = ray
        normal, offset = np.asarray(plane[0]), float(plane[1])
        denominator = normal @ direction
        if abs(denominator) < 1e-8:
            raise ValueError('ray is parallel to plane')
        distance = -(normal @ origin + offset) / denominator
        if distance <= 0:
            raise ValueError('intersection is behind camera')
        return origin + distance * direction

    def project(self, points):
        rotation, _ = cv2.Rodrigues(self.camera_from_base[:3, :3])
        image, _ = cv2.projectPoints(np.asarray(points, float), rotation, self.camera_from_base[:3, 3], self.intrinsic, self.distortion)
        return image.reshape(-1, 2)

    def estimate_cube_pose(self, detection, frame, support_z=None):
        if frame.camera_id != 'overview':
            raise ValueError('wrist top face has no absolute metric calibration')
        support_z = self.table_z if support_z is None else float(support_z)
        quad = np.asarray(detection['quad'], float)
        if self.known_cube is not None:
            side = float(self.known_cube['cube_size_m'])
            points = np.array([self.intersect_ray_with_plane(self.pixel_ray_in_base(p,frame),([0,0,1],-support_z-side)) for p in quad])
            edges = np.roll(points[:,:2],-1,axis=0)-points[:,:2]
            lengths = np.linalg.norm(edges,axis=1)
            normalized = edges/lengths[:,None]
            orthogonal_error = float(np.max(np.abs(np.sum(normalized*np.roll(normalized,-1,axis=0),axis=1))))
            if np.any(lengths < .5*side) or np.any(lengths > 1.15*side) or orthogonal_error > .3:
                raise ValueError('colour region is inconsistent with a supported known upright cube')
            yaw = float(np.arctan2(edges[0,1],edges[0,0]))
            pose = np.eye(4);pose[:2,:2] = [[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]
            pose[:3,3] = [*points[:,:2].mean(0),support_z+side/2]
            sigma = float((self.calibration['reprojection_rmse_px']+1.5)*.65/self.intrinsic[0,0])
            return ObjectState('',detection['color'],detection['mask'],pose,np.full(3,side),detection['confidence'],frame.timestamp,
                evidence={'kind':'known_upright_cube','quad':quad.tolist(),'frame_id':frame.frame_id,
                    'dimension_source':self.known_cube['source_artifacts'],'side_sigma_m':.0015,
                    'position_sigma_m':[sigma,sigma,.002],'rotation_sigma_rad':[0.,0.,.035],
                    'support_z_m':support_z,'support_assumed':True,'metric_qualified':False,
                    'colour_patch_was_not_used_as_cube_size':True,'calibration_sha256':self.fingerprint})
        rough = np.array([self.intersect_ray_with_plane(self.pixel_ray_in_base(p, frame), ([0, 0, 1], -support_z)) for p in quad])
        signed_area = np.sum(rough[:,0]*np.roll(rough[:,1],-1)-rough[:,1]*np.roll(rough[:,0],-1))
        if signed_area < 0:
            quad,rough = quad[::-1],rough[::-1]
        edge = rough[1, :2] - rough[0, :2]
        initial = np.r_[rough[:, :2].mean(0), np.clip(np.linalg.norm(edge), .016, .06), np.arctan2(edge[1], edge[0])]
        square = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])

        def vertices(p):
            c, s = np.cos(p[3]), np.sin(p[3])
            xy = square * p[2] @ np.array([[c, s], [-s, c]]) + p[:2]
            return np.column_stack((xy, np.full(4, support_z + p[2])))

        def residual(p):
            return (self.project(vertices(p)) - quad).ravel()

        best = least_squares(residual,initial,
            bounds=(np.r_[initial[:2]-.03,.015,initial[3]-np.pi/2],np.r_[initial[:2]+.03,.065,initial[3]+np.pi/2]),max_nfev=60)
        rmse = float(np.sqrt(np.mean(best.fun**2)))
        if rmse > 2.5 or not .016 < best.x[2] < .064:
            raise ValueError(f'top face does not explain an upright cube: {rmse:.2f}px')
        x, y, side, yaw = best.x
        pose = np.eye(4)
        pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        pose[:3, 3] = [x, y, support_z+side/2.]
        # Reprojection and calibration noise are both present; one view does not
        # establish arbitrary tilt or support height.
        sigma = (self.calibration['reprojection_rmse_px'] + rmse + 1.) * .65 / self.intrinsic[0, 0]
        evidence = {'kind': 'upright_cube_top_face', 'rmse_px': rmse, 'position_sigma_m': sigma,
                    'support_z_m': support_z, 'support_assumed': True, 'frame_id': frame.frame_id,
                    'quad': quad.tolist(), 'calibration_sha256': self.fingerprint, 'metric_qualified': False}
        return ObjectState('', detection['color'], detection['mask'], pose, np.full(3, side),
                           min(detection['confidence'], max(0., 1.-rmse/8.)), frame.timestamp, evidence=evidence)

    def install_wrist_calibration(self, record, robot):
        from so101_pick_place.vision.wrist_camera_geometry import camera_projection
        if robot.last_state is None:
            raise ValueError('measured feedback is required to validate wrist calibration')
        camera_projection(record,robot.backend.model,robot.last_state['positions'])
        if record.get('independent_control',{}).get('passed') is not True:
            raise ValueError('independent live wrist control has not passed')
        self.wrist = dict(record)

    def estimate_object_from_views(self, observations):
        """Use native full-cube metric reconstruction for complete silhouettes."""
        from so101_pick_place.geometry.spatial_cube import fit_spatial_cube
        if any(item.get('mask_kind')!='complete_cube_silhouette' for item in observations):
            raise ValueError('top colour masks cannot be passed off as full cube silhouettes')
        return fit_spatial_cube([(item['mask'],item['projection']) for item in observations])

    def estimate_known_cube_from_views(self, obj, observations):
        """Fixed measured side, upright pose, actual calibrated simultaneous views.

        Z is estimated from pixels, rather than copied from a proposed stacking
        destination. Every view independently checks the fitted top face.
        """
        from itertools import permutations
        if len(observations) < 2 or not any(item['camera_id']=='wrist' for item in observations):
            raise ValueError('independent wrist and overview observations are required')
        side = float(obj.dimensions[0])
        square = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*side
        initial = np.r_[obj.pose[:3,3],np.arctan2(obj.pose[1,0],obj.pose[0,0])]

        def top(p):
            c,s = np.cos(p[3]),np.sin(p[3])
            xy = square@np.array([[c,s],[-s,c]])+p[:2]
            return np.column_stack((xy,np.full(4,p[2]+side/2)))

        prepared = []
        for item in observations:
            projection = item['projection']
            frame = item['frame']
            if not projection.get('validated'):
                raise ValueError('unvalidated metric camera projection')
            if item['camera_id']=='wrist' and (frame.joints is None or frame.motion_during_exposure_rad is None or frame.motion_during_exposure_rad > .002):
                raise ValueError('wrist exposure is not paired with stationary measured joints')
            intrinsic = np.asarray(projection['camera_matrix'])
            distortion = np.asarray(projection['distortion'])
            transform = np.asarray(projection['camera_from_robot'])
            rotation,_ = cv2.Rodrigues(transform[:3,:3])
            quad = np.asarray(item['quad'],float)
            if quad.shape != (4,2):
                raise ValueError('complete four-corner top face required')
            predicted,_ = cv2.projectPoints(top(initial),rotation,transform[:3,3],intrinsic,distortion)
            choices = [np.roll(q,k,axis=0) for q in (quad,quad[::-1]) for k in range(4)]
            ordered = min(choices,key=lambda q:np.linalg.norm(q-predicted.reshape(4,2)))
            prepared.append((rotation,transform[:3,3],intrinsic,distortion,ordered))

        def residual(p):
            return np.concatenate([(cv2.projectPoints(top(p),r,t,k,d)[0].reshape(4,2)-quad).ravel() for r,t,k,d,quad in prepared])
        result = least_squares(residual,initial,bounds=(np.r_[initial[:2]-.04,self.table_z+side/2-.003,initial[3]-np.pi/2],np.r_[initial[:2]+.04,.4,initial[3]+np.pi/2]),max_nfev=80)
        errors = residual(result.x).reshape(len(prepared),4,2)
        norms = np.linalg.norm(errors,axis=2)
        if np.any(np.sqrt(np.mean(norms**2,axis=1))>2.5) or norms.max()>5.:
            raise ValueError('known cube fails independent per-camera reprojection')
        covariance = np.linalg.pinv(result.jac.T@result.jac)*max(1.,float(np.mean(errors**2)))
        sigma = np.sqrt(np.maximum(0.,np.diag(covariance)))
        calibration_error = max(item['projection'].get('reprojection_rmse_px',0.) for item in observations)
        # Include measured reference-size and extrinsic errors, in addition to
        # the local image fit. Lack of a metric reference never becomes zero noise.
        metric_error = float(obj.evidence.get('side_sigma_m',.0015))+float(obj.evidence.get('extrinsic_sigma_m',.001))
        sigma[:3] = np.sqrt(sigma[:3]**2+metric_error**2)
        if sigma[:3].max()>.003 or sigma[3]>.04:
            raise ValueError('cube XYZ/orientation uncertainty exceeds grasp tolerance')
        pose = np.eye(4);x,y,z,yaw = result.x
        pose[:2,:2] = [[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]
        pose[:3,3] = [x,y,z]
        return ObjectState(obj.id,obj.color,obj.mask,pose,obj.dimensions.copy(),obj.confidence,
            max(item['frame'].timestamp for item in observations),evidence=obj.evidence|{
                'metric_qualified':True,'position_sigma_m':sigma[:3].tolist(),
                'rotation_sigma_rad':[0.,0.,float(sigma[3])],'reprojection_max_px':float(norms.max()),
                'geometry_source':'current calibrated wrist + overview top-face fit with measured side',
                'destination_height_was_not_assumed':True})

    def projection_for_frame(self, frame, robot):
        if frame.camera_id=='overview':
            return self.calibration
        if self.wrist is None or frame.joints is None or frame.gripper is None:
            raise ValueError('wrist projection lacks qualified hand-eye or measured exposure pose')
        if frame.motion_during_exposure_rad is None or frame.motion_during_exposure_rad > .002:
            raise ValueError('wrist moved during its exposure')
        from so101_pick_place.vision.wrist_camera_geometry import camera_projection
        return camera_projection(self.wrist,robot.backend.model,robot.to_native(frame.joints,frame.gripper))

    def localize_cube_centers(self, obj, observations, robot):
        """Triangulate the same coloured top centre; paper edges are not cube edges."""
        if len(observations)<2 or not any(f.camera_id=='wrist' for f,d in observations):
            raise ValueError('wrist and independent calibrated overview are required')
        rows,prepared,origins = [],[],[]
        for frame,detection in observations:
            projection = self.projection_for_frame(frame,robot)
            k,d = np.asarray(projection['camera_matrix']),np.asarray(projection['distortion'])
            camera = np.asarray(projection['camera_from_robot'])
            pixel = np.asarray(detection['center_px'],float)
            uv = cv2.undistortPoints(pixel.reshape(1,1,2),k,d).ravel()
            rows.extend((uv[0]*camera[2]-camera[0],uv[1]*camera[2]-camera[1]))
            origins.append(np.linalg.inv(camera)[:3,3])
            prepared.append((projection,pixel))
        if np.max(np.linalg.norm(np.asarray(origins)[:,None]-origins,axis=2))<.025:
            raise ValueError('insufficient independent camera baseline')
        _,_,vh = np.linalg.svd(rows)
        if abs(vh[-1,3])<1e-9:
            raise ValueError('rays do not determine a finite cube point')
        point = vh[-1,:3]/vh[-1,3]
        def residual(p):
            errors = []
            for projection,pixel in prepared:
                camera = np.asarray(projection['camera_from_robot'])
                rotation,_ = cv2.Rodrigues(camera[:3,:3])
                projected,_ = cv2.projectPoints(np.asarray(p)[None],rotation,camera[:3,3],np.asarray(projection['camera_matrix']),np.asarray(projection['distortion']))
                errors.extend(projected.ravel()-pixel)
            return np.asarray(errors)
        fit = least_squares(residual,point,max_nfev=30)
        errors = residual(fit.x).reshape(-1,2)
        if np.max(np.linalg.norm(errors,axis=1))>5.:
            raise ValueError('independent camera rays do not agree on cube centre')
        for projection,pixel in prepared:
            if (np.asarray(projection['camera_from_robot'])@np.r_[fit.x,1.])[2]<.025:
                raise ValueError('triangulated cube is behind a camera')
        singular = np.linalg.svd(fit.jac,compute_uv=False)
        if singular[-1]<1e-5*singular[0]:
            raise ValueError('cube depth is not independently observable')
        noise = max(1.,max(p['reprojection_rmse_px'] for p,_ in prepared))
        covariance = np.linalg.inv(fit.jac.T@fit.jac)*noise**2
        sigma = np.sqrt(np.diag(covariance)+float(obj.evidence.get('side_sigma_m',.0015))**2)
        if sigma.max()>.003:
            raise ValueError('triangulated cube uncertainty exceeds three millimetres')
        pose = obj.pose.copy();pose[:3,3] = fit.x-[0.,0.,obj.dimensions[2]/2]
        if pose[2,3]-obj.dimensions[2]/2<self.table_z-.003:
            raise ValueError('triangulated cube conflicts with calibrated table')
        return ObjectState(obj.id,obj.color,obj.mask,pose,obj.dimensions.copy(),obj.confidence,
            max(frame.timestamp for frame,d in observations),evidence=obj.evidence|{
              'metric_qualified':True,'position_sigma_m':sigma.tolist(),'geometry_source':'calibrated wrist/overview top-centre rays',
              'destination_height_was_not_assumed':True,'top_point_m':fit.x.tolist(),'per_view_error_px':np.linalg.norm(errors,axis=1).tolist()})

    @staticmethod
    def object_pose_to_tool_pose(object_pose, held_transform):
        return np.asarray(object_pose) @ np.linalg.inv(held_transform)


class AttachmentGeometry:
    """Qualified rigid cells and a measured conservative cable swept volume."""
    def __init__(self, profile_path, evidence_path, calibration_sha256):
        profile_path,evidence_path = Path(profile_path),Path(evidence_path)
        profile = json.loads(profile_path.read_text())
        evidence = json.loads(evidence_path.read_text())
        fingerprint = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        if evidence.get('attachment_sha256') != fingerprint or evidence.get('overview_calibration_sha256') != calibration_sha256:
            raise ValueError('attachment evidence does not match current geometry/calibration')
        if evidence.get('cable_sweep_verified') is not True or evidence.get('accessory_geometry_verified') is not True:
            raise ValueError('camera, mount and cable sweep are not independently qualified')
        views = evidence.get('independent_views',[])
        if len(views) < 3 or max(float(v.get('reprojection_error_px',99)) for v in views) > 5.:
            raise ValueError('three independent measured attachment views within 5px are required')
        if evidence.get('holdout_control',{}).get('passed') is not True:
            raise ValueError('independent attachment holdout control is missing')
        for view in views:
            if not Path(view.get('image_path','')).is_file() or abs(float(view.get('image_timestamp',0))-float(view.get('feedback_timestamp',1))) > .1:
                raise ValueError('attachment image is missing or not paired with fresh measured joints')
        self.domain = np.asarray(evidence.get('joint_domain_rad'),float)
        if self.domain.shape != (5,2) or not np.isfinite(self.domain).all() or np.any(self.domain[:,0] >= self.domain[:,1]):
            raise ValueError('measured cable qualification must bound its joint domain')
        parts = [*profile.get('parts',[]),*profile.get('cable_parts',[])]
        if not profile.get('cable_parts'):
            raise ValueError('cable swept geometry is missing')
        self.parts = []
        for part in parts:
            lo,hi = np.asarray(part['lower_m'],float),np.asarray(part['upper_m'],float)
            if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo,hi]).all() or np.any(lo>=hi):
                raise ValueError('invalid accessory cell')
            points = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
            self.parts.append((part['name'],part['parent_link'],lo,hi,points))
        self.fingerprint = fingerprint

    def reason(self, robot, joints, scene, target=None, payload_local=None, placed_cube=None):
        from dataclasses import replace
        from so101_pick_place.experiments.box_extraction.mesh_collision import mesh_box_overlap,load_collision_meshes
        from so101_pick_place.experiments.box_extraction.scene import obb_overlap
        q = robot.from_native(joints)
        if np.any(q < self.domain[:,0]) or np.any(q > self.domain[:,1]):
            return 'pose leaves measured camera/cable qualification domain'
        frames = robot.backend.model.frames(joints)
        meshes = load_collision_meshes(str(robot.root/'vendor/SO101/so101_new_calib.urdf'))
        for name,parent,lo,hi,local in self.parts:
            transform = frames[parent]
            points = local @ transform[:3,:3].T + transform[:3,3]
            if points[:,2].min() < robot.backend.model.table_z+.004:
                return name+': table clearance'
            wall = scene.box.collision(points)
            if wall:
                return name+': '+wall
            for link,mesh in meshes.items():
                # Only the fastening parent is an intended mechanical contact.
                if link != parent and mesh_box_overlap(mesh,frames[link],transform,lo,hi):
                    return name+': robot link '+link
            for object_id,cube in scene.cubes.items():
                if object_id==target and placed_cube is not None:
                    cube = placed_cube
                if object_id==target and payload_local is not None:
                    tool = frames['gripper_frame_link']
                    vertices = np.asarray(payload_local) @ tool[:3,:3].T + tool[:3,3]
                    if np.all(points.max(0) >= vertices.min(0)) and np.all(points.min(0) <= vertices.max(0)):
                        return name+': carried cube'
                elif obb_overlap(points,cube):
                    return name+': cube '+object_id
        return None
