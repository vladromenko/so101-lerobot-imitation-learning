"""Calibrated overview rays and explicit upright-cube top-face fitting."""
import hashlib
import json
import time
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
        self.attachment_projection_cache = None
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
                    'support_z_m':support_z,'support_assumed':True,
                    'metric_qualified':getattr(self,'grasp_mode',None)=='overhead_grasp' and
                        bool(np.all(lengths>.85*side)) and orthogonal_error<.15,
                    'geometry_mode':getattr(self,'grasp_mode','wrist_refinement'),
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

    def measure_visible_attachment(self, frame, robot, hypothesis, *, now=None):
        """Use the calibrated fixed camera without moving or certifying a hidden gap."""
        from .vision import visible_attachment_features
        from so101_pick_place.vision.calibration_control import validate_overview_control
        if frame.camera_id!='overview' or frame.joints is None or frame.gripper is None:
            raise ValueError('attachment measurement requires exposure-paired encoder feedback')
        age=(time.time() if now is None else now)-frame.capture_interval[0]
        if not 0.<=age<=.5:
            raise ValueError('attachment measurement frame is stale')
        if frame.motion_during_exposure_rad is None or frame.motion_during_exposure_rad>.002:
            raise ValueError('attachment exposure was not stationary')
        q = robot.to_native(frame.joints,frame.gripper)
        frames = robot.backend.model.frames(q)
        part = next((p for p in hypothesis['parts'] if p['name']=='wrist_camera_board_lens'),None)
        if part is None:
            raise ValueError('archived camera-board geometry is unavailable')
        lo,hi = np.asarray(part['lower_m']),np.asarray(part['upper_m'])
        corners = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        transform = frames[part['parent_link']]
        points = corners@transform[:3,:3].T+transform[:3,3]
        projected = self.project(points)
        visible = visible_attachment_features(frame,projected)
        mask = visible.pop('board_mask')
        key=(tuple(q.items()),mask.shape,self.fingerprint,tuple(lo),tuple(hi),part['parent_link'],tuple(robot.backend.model.offsets))
        cached=self.attachment_projection_cache
        if cached is None or cached[0]!=key:
            arm = np.zeros(mask.shape,np.uint8)
            from so101_pick_place.experiments.box_extraction.mesh_collision import load_collision_meshes
            meshes = load_collision_meshes(str(robot.root/'vendor/SO101/so101_new_calib.urdf'))
            minimum_depth = float('inf')
            for link in ('upper_arm_link','lower_arm_link','wrist_link'):
                mesh = np.vstack([p.triangles.reshape(-1,3) for p in meshes[link]])
                vertices = mesh@frames[link][:3,:3].T+frames[link][:3,3]
                minimum_depth=min(minimum_depth,float((vertices@self.camera_from_base[:3,:3].T+self.camera_from_base[:3,3])[:,2].min()))
                cv2.fillConvexPoly(arm,cv2.convexHull(np.int32(self.project(vertices))),255)
            distance = cv2.distanceTransform(255-arm,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            self.attachment_projection_cache=key,arm,distance,minimum_depth
        else:
            _,arm,distance,minimum_depth=cached
        gap = float(distance[mask>0].min()) if np.any(mask) else None
        predicted = np.zeros(mask.shape,np.uint8)
        cv2.fillConvexPoly(predicted,cv2.convexHull(np.int32(projected)),255)
        overlap = np.column_stack(np.where((predicted>0)&(arm>0)))[:,::-1]
        unresolved = projected if not len(overlap) else overlap
        region = np.r_[unresolved.min(0),unresolved.max(0)].astype(int).tolist()
        if visible['board_hull_px'] is not None:
            outline=np.asarray(visible['board_hull_px'])
            region=np.r_[outline.min(0),outline.max(0)].astype(int).tolist()
        try:
            control = validate_overview_control(frame.image,self.calibration,robot.backend.model,q)
        except ValueError as exc:
            control = {'passed':False,'reason':str(exc),'parameters_changed':False}
        visible_bound = None
        if gap is not None and control['passed'] and np.allclose(self.distortion,0.) and minimum_depth>0:
            pixels=np.column_stack(np.where(mask>0))[:,::-1]
            h,w=mask.shape
            edge=float(min(pixels[:,0].min(),pixels[:,1].min(),w-1-pixels[:,0].max(),h-1-pixels[:,1].max()))
            padding=control['error_px']+3*self.calibration['reprojection_rmse_px']+1.
            sensor=np.array([[0.,0.],[w,0.],[w,h],[0.,h]])
            radius=np.max(np.linalg.norm((sensor-self.intrinsic[:2,2])/self.intrinsic[[0,1],[0,1]],axis=1))
            angle=max(0.,min(gap,edge)-padding)/(max(self.intrinsic[0,0],self.intrinsic[1,1])*(1+radius**2))
            visible_bound={'lower_bound_m':float(minimum_depth*np.sin(angle)),
                'image_uncertainty_padding_px':float(padding),
                'method':'calibrated-ray angular separation from complete triangle projections of the compared links',
                'arm_links_compared':['upper_arm_link','lower_arm_link','wrist_link'],
                'scope':'visible coloured board surface versus the listed links at current stationary pose only; hidden surfaces, other objects and cable are excluded'}
        optical = points@self.camera_from_base[:3,:3].T+self.camera_from_base[:3,3]
        depth = [float(optical[:,2].min()-.003),float(optical[:,2].max()+.003)]
        bounds = None
        if visible['board_hull_px'] is not None:
            pixels = np.asarray(visible['board_hull_px'],float)
            rays = cv2.undistortPoints(pixels.reshape(-1,1,2),self.intrinsic,self.distortion).reshape(-1,2)
            pose = np.linalg.inv(self.camera_from_base)
            cloud = np.vstack([np.column_stack((rays,np.ones(len(rays))))*z for z in depth])
            cloud = cloud@pose[:3,:3].T+pose[:3,3]
            bounds = {'lower_m':cloud.min(0).tolist(),'upper_m':cloud.max(0).tolist(),
                'condition':'depth bracket from archived CAD placement; not independently measured depth'}
        return visible|{'frame_id':frame.frame_id,'capture_interval':list(frame.capture_interval),
            'feedback_error_s':frame.feedback_error_s,'joints_rad':frame.joints.tolist(),
            'gripper_native':float(frame.gripper),
            'overview_fk_control':control,'tool_pose_base_m':frames['gripper_frame_link'].tolist(),
            'tool_pose_camera_m':(self.camera_from_base@frames['gripper_frame_link']).tolist(),
            'board_bounds_base_conditional':bounds,'depth_bracket_cad_only_m':depth,
            'board_to_arm_projected_gap_px':gap,'physical_gap_m':None,
            'visible_surface_clearance_bound':visible_bound,
            'pnp_status':'not identifiable: image features have no independently known 3D landmark correspondences',
            'metric_depth_observable':False,'independent_view_count':1,
            'image_region_is_visible_reference_only':True,
            'unresolved_region_px':region,'unresolved_part':'underside of camera support; the visible USB wire exit is on its outside/right; wrist camera cannot see its own rear',
            'needed_measurement':'minimum physical clearance of that hidden assembly to the arm over the initial wrist segment, including the tracking-error envelope',
            'motion_authorized':False,'calibration_changed':False,'cable_3d_pose_inferred':False}

    def rigid_pose_from_landmarks(self, points_m, pixels, frame, robot, landmark_sigma_m):
        """PnP for corresponding physical landmarks, not a segmentation bounding box."""
        points,pixels = np.asarray(points_m,float),np.asarray(pixels,float)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or pixels.shape!=(len(points),2):
            raise ValueError('four or more paired metric physical landmarks are required')
        if not np.isfinite(points).all() or not np.isfinite(pixels).all() or landmark_sigma_m<=0:
            raise ValueError('finite landmarks and their measured uncertainty are required')
        if np.linalg.matrix_rank(points-points.mean(0))<2 or len(np.unique(points,axis=0))!=len(points):
            raise ValueError('landmarks are repeated or collinear')
        projection = self.projection_for_frame(frame,robot)
        k,d = np.asarray(projection['camera_matrix']),np.asarray(projection['distortion'])
        solutions = cv2.solvePnPGeneric(points,pixels,k,d,flags=cv2.SOLVEPNP_SQPNP)
        accepted = []
        for rotation,translation in zip(solutions[1],solutions[2]):
            matrix,_ = cv2.Rodrigues(rotation)
            projected,jacobian = cv2.projectPoints(points,rotation,translation,k,d)
            errors = np.linalg.norm(projected.reshape(-1,2)-pixels,axis=1)
            if np.min((points@matrix.T+translation.ravel())[:,2])>.025 and errors.max()<=5.:
                accepted.append((float(np.sqrt(np.mean(errors**2))),matrix,translation.ravel(),jacobian[:,:6]))
        accepted.sort(key=lambda item:item[0])
        if not accepted or accepted[0][0]>2.5:
            raise ValueError('physical landmarks do not support a cheiral PnP pose')
        if len(accepted)>1 and accepted[1][0]-accepted[0][0]<1.:
            raise ValueError('PnP pose remains ambiguous; another checked view is required')
        error,rotation,translation,jacobian = accepted[0]
        if np.linalg.matrix_rank(jacobian)<6:
            raise ValueError('PnP pose is not observable')
        covariance = np.linalg.inv(jacobian.T@jacobian)*max(1.,error**2)
        diameter = float(np.linalg.norm(np.ptp(points,axis=0)))
        scale_error = landmark_sigma_m*np.linalg.norm(translation)/max(diameter,1e-6)
        base_from_camera=np.linalg.inv(np.asarray(projection['camera_from_robot']))
        translation_covariance=base_from_camera[:3,:3]@(covariance[3:6,3:6]+np.eye(3)*scale_error**2)@base_from_camera[:3,:3].T
        sigma = np.sqrt(np.diag(translation_covariance))
        camera_from_object = np.eye(4);camera_from_object[:3,:3]=rotation;camera_from_object[:3,3]=translation
        return {'base_from_object':(base_from_camera@camera_from_object).tolist(),
            'translation_sigma_m':sigma.tolist(),'reprojection_rmse_px':error,'frame_id':frame.frame_id,
            'method':'metric physical-landmark PnP','calibration_changed':False}

    def reconstruct_feature_cloud(self, observations, robot):
        """Native descriptor correspondence + independent metric rays, with covariance."""
        from so101_pick_place.vision.wrist_feature_calibration import feature_tracks
        from so101_pick_place.experiments.box_extraction.perception import triangulate_track
        if len(observations)<3 or len({item['frame'].frame_id for item in observations})!=len(observations):
            raise ValueError('three distinct physical observations are required')
        now=time.time()
        if any(not 0.<=now-item['frame'].timestamp<=60. for item in observations) or now-max(item['frame'].timestamp for item in observations)>.5:
            raise ValueError('feature reconstruction requires a current exposure and a bounded observation history')
        projections = [self.projection_for_frame(item['frame'],robot) for item in observations]
        origins=np.asarray([np.linalg.inv(np.asarray(p['camera_from_robot']))[:3,3] for p in projections])
        if np.max(np.linalg.norm(origins[:,None]-origins[None,:],axis=2))<.025:
            raise ValueError('repeated fixed-camera images provide no metric depth parallax')
        tracks = feature_tracks([item['frame'].image for item in observations],[item['mask'] for item in observations])
        cloud,rejected = [],[]
        for track in tracks:
            try:
                point = triangulate_track(track['measurements'],projections)
                rows = []
                for measurement in track['measurements']:
                    p = projections[measurement['view']]
                    rows.extend(self.point_image_jacobian(point,p).tolist())
                matrix = np.asarray(rows)
                noise = max(1.,max(p.get('reprojection_rmse_px',0.) for p in projections))
                covariance = np.linalg.inv(matrix.T@matrix)*noise**2
                cloud.append({'point_m':point.tolist(),'sigma_m':np.sqrt(np.diag(covariance)).tolist(),
                    'measurements':track['measurements']})
            except (ValueError,np.linalg.LinAlgError) as exc:
                rejected.append(str(exc))
        if len(cloud)<12:
            raise ValueError('fewer than twelve independently reconstructed features: '+str(rejected[:2]))
        return {'points':cloud,'rejected_tracks':len(rejected),'metric_depth_measured':True,
            'frame_ids':[item['frame'].frame_id for item in observations]}

    @staticmethod
    def point_image_jacobian(point, projection):
        camera = np.asarray(projection['camera_from_robot']);rotation,_=cv2.Rodrigues(camera[:3,:3])
        k,d=np.asarray(projection['camera_matrix']),np.asarray(projection['distortion'])
        def project(p):
            return cv2.projectPoints(np.asarray(p)[None],rotation,camera[:3,3],k,d)[0].ravel()
        step=1e-5
        return np.column_stack([(project(point+axis*step)-project(point-axis*step))/(2*step) for axis in np.eye(3)])


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
        self.allowed_contacts = {}

    @classmethod
    def attended(cls, profile_path, robot, observation):
        """Historical rigid CAD, with the cable explicitly watched by a human.

        This is not independent cable qualification and is never persisted as
        one. Full rigid collision checks remain active on the planned path.
        """
        if observation.get('cable_and_hidden_surfaces_clear') is not True:
            raise ValueError('attended overhead operation requires the operator observation')
        obj = cls.__new__(cls)
        path = Path(profile_path)
        profile = json.loads(path.read_text())
        obj.parts,obj.allowed_contacts = [],{}
        for part in profile['parts']:
            lo,hi = np.asarray(part['lower_m']),np.asarray(part['upper_m'])
            points = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
            obj.parts.append((part['name'],part['parent_link'],lo,hi,points))
            obj.allowed_contacts[part['name']] = set(part.get('allow_contact_links',[part['parent_link']]))
        obj.domain = np.deg2rad(np.asarray(robot.backend.model.ranges,float))
        obj.fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        obj.operator_observation = dict(observation)
        return obj

    def reason(self, robot, joints, scene, target=None, payload_local=None, placed_cube=None):
        from dataclasses import replace
        from so101_pick_place.experiments.box_extraction.mesh_collision import mesh_box_overlap,load_collision_meshes
        from so101_pick_place.experiments.box_extraction.scene import obb_overlap
        q = robot.from_native(joints)
        if np.any(q < self.domain[:,0]) or np.any(q > self.domain[:,1]):
            return 'pose leaves measured camera/cable qualification domain'
        frames = robot.backend.model.frames(joints)
        meshes = load_collision_meshes(str(robot.root/'vendor/SO101/so101_new_calib.urdf'))
        if not hasattr(self,'link_bounds'):
            self.link_bounds = {}
            for link,parts in meshes.items():
                lo = np.min([part.tree[0] for part in parts],axis=0)
                hi = np.max([part.tree[1] for part in parts],axis=0)
                self.link_bounds[link] = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        relative = {}
        for parent in {part[1] for part in self.parts}:
            inverse = np.linalg.inv(frames[parent])
            relative[parent] = {}
            for link,bounds in self.link_bounds.items():
                points = robot.backend._transform_points(inverse@frames[link],bounds)
                relative[parent][link] = points.min(0),points.max(0)
        for name,parent,lo,hi,local in self.parts:
            transform = frames[parent]
            points = local @ transform[:3,:3].T + transform[:3,3]
            if points[:,2].min() < robot.backend.model.table_z+.004:
                return name+': table clearance'
            wall = scene.box.collision(points)
            if wall:
                recovery = getattr(self,'recovery_walls',{})
                if not recovery:
                    return name+': '+wall
                inverse = np.linalg.inv(scene.box.base_from_box)
                local_points = points@inverse[:3,:3].T+inverse[:3,3]
                for lower,upper,label in scene.box.solid_parts():
                    if np.all(local_points.max(0)>=lower) and np.all(local_points.min(0)<=upper):
                        bound = recovery.get((name,label))
                        if bound is None:
                            return name+': '+label
                        gap = float(np.max(np.maximum(bound['lower']-local_points.max(0),local_points.min(0)-bound['upper'])))
                        if gap<bound['minimum']:
                            return name+': recovery decreases '+label
            for link,mesh in meshes.items():
                # Only the fastening parent is an intended mechanical contact.
                intended = link==parent or link in self.allowed_contacts.get(name,set())
                lower,upper = relative[parent][link]
                separated = np.any(upper<lo-1e-9) or np.any(lower>hi+1e-9)
                if not intended and not separated and mesh_box_overlap(mesh,frames[link],transform,lo,hi):
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
