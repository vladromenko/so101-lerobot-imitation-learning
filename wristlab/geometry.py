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

    def camera_pose_in_base(self, q, camera_id, robot=None):
        if camera_id == 'overview':
            return np.linalg.inv(self.camera_from_base)
        if self.wrist is None or robot is None:
            raise ValueError('wrist-to-link calibration has not passed independent control')
        return robot.link_pose(q, self.wrist['parent_link']) @ np.asarray(self.wrist['link_from_camera'])

    def pixel_ray_in_base(self, pixel, frame):
        if frame.camera_id != 'overview':
            raise ValueError('no qualified metric wrist ray; use image-space servoing')
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

    def reason(self, robot, joints, scene, target=None, payload_local=None):
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
                if object_id==target and payload_local is not None:
                    tool = frames['gripper_frame_link']
                    vertices = np.asarray(payload_local) @ tool[:3,:3].T + tool[:3,3]
                    if np.all(points.max(0) >= vertices.min(0)) and np.all(points.min(0) <= vertices.max(0)):
                        return name+': carried cube'
                elif obb_overlap(points,cube):
                    return name+': cube '+object_id
        return None
