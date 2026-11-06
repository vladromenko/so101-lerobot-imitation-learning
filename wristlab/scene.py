"""Observed identities and reservations. Unmeasured geometry stays unqualified."""
from dataclasses import replace
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from so101_pick_place.geometry.spatial_cube import SpatialCube
from .types import SceneState, BoxState, Placement
from .vision import detect_cubes, detect_boxes


class Scene:
    def __init__(self, geometry, robot=None):
        self.geometry, self.robot = geometry, robot
        self.state = SceneState()
        self.next_id = 1
        self.reserved, self.reserved_slots = set(), set()
        self.held = None
        self.raw = {}
        self.frames = {}

    def update_scene(self, frames):
        frame = frames['overview']
        detected = detect_cubes(frame)
        objects, rejected = [], []
        for detection in detected:
            try:
                objects.append(self.geometry.estimate_cube_pose(detection, frame))
            except ValueError as exc:
                rejected.append({'color': detection['color'], 'reason': str(exc)})
        old = list(self.state.objects.values())
        if old and objects:
            costs = np.array([[np.linalg.norm(a.pose[:3,3]-b.pose[:3,3]) + (0. if a.color == b.color else 10.) for b in objects] for a in old])
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row,column] < .04:
                    objects[column].id = old[row].id
        for obj in objects:
            if not obj.id:
                obj.id = f'cube-{self.next_id:03d}'
                self.next_id += 1
            if obj.id in self.reserved:
                obj.status = 'reserved'
        # Edge proposals are expensive relative to HSV; refresh them once a second.
        if not self.state.boxes or frame.timestamp-self.state.evidence.get('boxes_at',0.) > 5.:
            proposals = detect_boxes(frame)
            if self.robot is not None and self.robot.last_state is not None:
                from box_extract.scene_builder import robot_mask
                mask = robot_mask(self.robot.backend,self.robot.last_state['positions'],self.geometry.calibration,frame.image.shape[:2])
                retained = []
                for proposal in proposals:
                    region = np.zeros(mask.shape,np.uint8)
                    cv2.fillPoly(region,[np.array(proposal['polygon'],np.int32)],1)
                    overlap = np.count_nonzero((region > 0) & (mask > 0))/max(1,np.count_nonzero(region))
                    if overlap < .05:
                        retained.append(proposal)
                proposals = retained
            boxes = {}
            for i, proposal in enumerate(proposals):
                center = self.geometry.intersect_ray_with_plane(self.geometry.pixel_ray_in_base(proposal['center_px'], frame), ([0,0,1],-self.geometry.table_z))
                pose = np.eye(4);pose[:3,3] = center
                box = BoxState(f'box-{i+1}',pose,np.array([np.nan,np.nan,np.nan]),None,evidence=proposal | {'metric_qualified':False})
                boxes[box.id] = box
            boxes_at = frame.timestamp
        else:
            boxes, boxes_at = self.state.boxes, self.state.evidence['boxes_at']
        for box in boxes.values():
            polygon = np.array(box.evidence['polygon'],np.float32)
            box.occupied_slots = [o.id for o in objects if cv2.pointPolygonTest(polygon,tuple(np.mean(o.evidence['quad'],axis=0)),False) > 0]
        self.state = SceneState({o.id:o for o in objects},boxes,rejected,
            [{'id':'table','height_m':self.geometry.table_z,'source':'native calibrated plane'}],self.state.version+1,frame.timestamp,
            {'boxes_at':boxes_at,'coverage_verified':False,'metric_camera':'overview','wrist_metric_qualified':False})
        self.raw = {'overview':detected,'wrist':detect_cubes(frames['wrist'])}
        self.frames = dict(frames)
        return self.state

    def reserve_object(self, object_id):
        if object_id in self.reserved or self.held is not None:
            raise ValueError('object is reserved or gripper already holds a payload')
        obj = self.state.objects[object_id]
        if time.time()-obj.timestamp > 1.:
            raise ValueError('target observation is stale')
        self.reserved.add(object_id)
        return obj

    def release_reservation(self, object_id):
        self.reserved.discard(object_id)
        self.reserved_slots = {s for s in self.reserved_slots if s[0] != object_id}

    def attach_object(self, object_id, transform):
        self.held = (object_id,np.asarray(transform))
        self.state.objects[object_id].status = 'held'

    def detach_object(self, object_id, observed_pose):
        self.state.objects[object_id].pose = np.asarray(observed_pose)
        self.state.objects[object_id].status = 'observed'
        self.held = None
        self.release_reservation(object_id)

    def find_free_box_slots(self, box_id, object_shape):
        box = self.state.boxes[box_id]
        if not box.evidence.get('metric_qualified'):
            raise ValueError('box opening, floor and wall height have no qualified metric survey')
        from box_extract.stow import box_floor_patches
        native = box.evidence['native_scene']
        from box_extract.scene_builder import SceneParameters
        return box_floor_patches(native,SceneParameters())

    def find_free_table_slots(self, object_shape):
        return [s for s in self.state.surfaces if s.get('free_surface_verified') and s.get('patch') is not None]

    def build_support_graph(self):
        objects = list(self.state.objects.values())
        graph = {o.id:[] for o in objects}
        for lower in objects:
            for upper in objects:
                dz = upper.pose[2,3]-lower.pose[2,3]
                gap = abs(dz-(lower.dimensions[2]+upper.dimensions[2])/2)
                xy = np.abs(upper.pose[:2,3]-lower.pose[:2,3])
                if lower.id != upper.id and dz > 0 and gap < .004 and np.all(xy < (lower.dimensions[:2]+upper.dimensions[:2])/2):
                    graph[lower.id].append(upper.id)
        return graph

    def native_object(self, object_id):
        from so101_pick_place.geometry.cuboid_vision import cube_vertices
        o = self.state.objects[object_id]
        if not o.evidence.get('metric_qualified'):
            raise ValueError('cube metric pose/support has not passed independent validation')
        yaw = np.arctan2(o.pose[1,0],o.pose[0,0])
        vertices = cube_vertices(o.pose[:2,3],float(o.dimensions[0]),yaw,o.pose[2,3]-o.dimensions[2]/2)
        return SpatialCube(o.pose[:3,3],float(o.dimensions[0]),o.pose[:3,:3],vertices,o.evidence)

    def public(self):
        return {'version':self.state.version,'timestamp':self.state.timestamp,
            'cubes':[{'id':o.id,'color':o.color,'px':np.mean(o.evidence['quad'],axis=0).tolist(),'xy_m':o.pose[:2,3].tolist(),
                      'dimensions_m':o.dimensions.tolist(),'confidence':o.confidence,'polygon':[o.evidence['quad']],
                      'inside':any(o.id in b.occupied_slots for b in self.state.boxes.values()),'status':o.status,
                      'metric_qualified':o.evidence['metric_qualified']} for o in self.state.objects.values()],
            'boxes':[{'id':b.id,'polygon':b.evidence['polygon'],'occupied':b.occupied_slots,'metric_qualified':b.evidence['metric_qualified']} for b in self.state.boxes.values()],
            'inside':sum(any(o.id in b.occupied_slots for b in self.state.boxes.values()) for o in self.state.objects.values()),
            'outside':sum(not any(o.id in b.occupied_slots for b in self.state.boxes.values()) for o in self.state.objects.values()),
            'opening':None,'time':time.time(),'evidence':self.state.evidence,'rejected':self.state.obstacles}

    def native_scene(self, object_id, placement):
        """Translate a qualified survey into the existing hollow-wall world."""
        from so101_pick_place.experiments.box_extraction.scene import ExtractionScene
        surface = next((s for s in self.state.surfaces if s['id']==placement.support_surface),None)
        if placement.support_surface in self.state.boxes:
            box = self.state.boxes[placement.support_surface]
            if not box.evidence.get('metric_qualified'):
                raise ValueError('box floor/rim/walls are not independently measured')
            patch = box.evidence['support_patch']
            primary = box.evidence['native_box']
        elif surface is not None and surface.get('patch') is not None:
            patch = surface['patch']
            qualified = [b for b in self.state.boxes.values() if b.evidence.get('native_box') is not None]
            if not qualified:
                raise ValueError('all visible boxes must have measured collision geometry')
            primary = qualified[0].evidence['native_box']
        else:
            raise ValueError('placement has no surveyed support surface')
        if any(not b.evidence.get('metric_qualified') for b in self.state.boxes.values()):
            raise ValueError('an unmeasured second box remains in the workspace')
        if not self.state.evidence.get('coverage_verified'):
            raise ValueError('workspace coverage and unknown obstacles are not qualified')
        cubes = {key:self.native_object(key) for key in self.state.objects}
        boxes = [b.evidence['native_box'] for b in self.state.boxes.values()]
        return ExtractionScene(str(self.state.version),self.state.timestamp,BoxUnion(primary,boxes),cubes,patch,(),True)


class BoxUnion:
    """Keep every measured box hollow, expressed in the active box frame."""
    def __init__(self, primary, boxes):
        self.primary, self.boxes = primary, boxes

    def __getattr__(self, name):
        return getattr(self.primary,name)

    def solid_parts(self):
        inverse = np.linalg.inv(self.primary.base_from_box)
        result = []
        for box in self.boxes:
            transform = inverse @ box.base_from_box
            for lo,hi,label in box.solid_parts():
                corners = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
                points = corners @ transform[:3,:3].T + transform[:3,3]
                name = label if box is self.primary else 'other box: '+label
                result.append((points.min(0),points.max(0),name))
        return result

    def collision(self, points):
        return next((reason for box in self.boxes if (reason:=box.collision(points)) is not None),None)
