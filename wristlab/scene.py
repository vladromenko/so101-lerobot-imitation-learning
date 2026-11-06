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
        self.pending_placement = None
        self.raw = {}
        self.frames = {}
        self.track_memory = {}
        self.reset_targets = {}
        self.metric_objects = {}
        self.metric_boxes = {}
        self.measured_surfaces = []
        self.boxes_semantically_seeded = False
        self.attachment_hypothesis = None
        self.passive_measurements = None

    def install_box_proposals(self, scan):
        """One native semantic fallback at startup; track its actual source image."""
        from pathlib import Path
        from .types import Frame
        image = cv2.imread(str(Path(scan['artifacts'])/'frame.jpg'))
        if image is None:
            raise ValueError('native box observation image is missing')
        captured = float(scan['frame']['captured_at_unix_s'])
        if not 0. <= time.time()-captured <= 30.:
            raise ValueError('native box observation is stale')
        proposals = [d for d in scan['detections'] if d['label']=='box' and d['score']>=.7]
        if not proposals:
            raise ValueError('native fallback found no unambiguous box bodies')
        boxes = {}
        for index,detected in enumerate(sorted(proposals,key=lambda d:d['center_px'][0])):
            points = np.vstack(detected['silhouette_polygons_px']).astype(np.float32)
            polygon = cv2.convexHull(points).reshape(-1,2)
            key = 'box-'+str(index+1)
            boxes[key] = BoxState(key,np.eye(4),np.full(3,np.nan),None,evidence={
                'polygon':polygon.tolist(),'center_px':polygon.mean(0).tolist(),'metric_qualified':False,
                'opening_verified':False,'height_verified':False,'association_source':'native semantic box body, optical-flow tracked',
                'source_timestamp':captured})
        self.state.boxes = boxes
        self.state.timestamp = captured
        self.state.evidence['boxes_at'] = captured
        self.frames['overview'] = Frame('overview','native:'+str(scan['frame']['sequence']),image,captured,time.time(),(captured,captured))
        self.boxes_semantically_seeded = True

    def track_box_edges(self, frame):
        previous = self.frames.get('overview')
        if previous is None or not self.state.boxes:
            return None
        gray0 = cv2.cvtColor(previous.image,cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame.image,cv2.COLOR_BGR2GRAY)
        proposed = []
        for box in self.state.boxes.values():
            polygon = np.asarray(box.evidence['polygon'],np.float32)
            mask = np.zeros(gray0.shape,np.uint8)
            cv2.polylines(mask,[np.int32(polygon)],True,255,18)
            points = cv2.goodFeaturesToTrack(gray0,40,.01,6,mask=mask)
            if points is None or len(points)<6:
                return None
            tracked,status,error = cv2.calcOpticalFlowPyrLK(gray0,gray1,points,None,winSize=(21,21),maxLevel=2)
            good = status.ravel()>0
            if good.sum()<6:
                return None
            transform,inliers = cv2.estimateAffinePartial2D(points[good],tracked[good],method=cv2.RANSAC,ransacReprojThreshold=2.)
            if transform is None or inliers.sum()<6:
                return None
            if np.max(np.abs(transform[:,:2]-np.eye(2)))>.03 or np.linalg.norm(transform[:,2])>20.:
                return None
            corners = polygon@transform[:,:2].T+transform[:,2]
            proposed.append(dict(box.evidence,polygon=corners.tolist(),center_px=corners.mean(0).tolist(),
                                 line_tracking_inliers=int(inliers.sum())))
        return proposed

    def update_scene(self, frames):
        frame = frames['overview']
        detected = detect_cubes(frame)
        wrist_detections = detect_cubes(frames['wrist'])
        objects, rejected = [], []
        for detection in detected:
            try:
                support_z = self.geometry.table_z
                for box in self.state.boxes.values():
                    if box.evidence.get('metric_qualified') and cv2.pointPolygonTest(np.asarray(box.evidence['polygon'],np.float32),tuple(detection['center_px']),False)>0:
                        measured = box.evidence['native_box']
                        support_z = float((measured.base_from_box@np.r_[0.,0.,measured.floor_interval_m[1],1.])[2])
                obj = self.geometry.estimate_cube_pose(detection,frame,support_z)
                wrist = [d for d in wrist_detections if d['color']==obj.color]
                if self.geometry.wrist is not None and self.robot is not None and len(wrist)==1:
                    try:
                        obj = self.geometry.localize_cube_centers(obj,[(frame,detection),(frames['wrist'],wrist[0])],self.robot)
                    except ValueError as exc:
                        obj.evidence['wrist_measurement_rejected'] = str(exc)
                objects.append(obj)
            except ValueError as exc:
                rejected.append({'color': detection['color'], 'reason': str(exc)})
        for obj in objects:
            matches = [] if obj.evidence.get('metric_qualified') else [(key,cube) for key,cube in self.metric_objects.items() if np.linalg.norm(cube.center_m[:2]-obj.pose[:2,3]) < .006]
            if len(matches)==1:
                key,cube = matches[0]
                projected = self.geometry.project(cube.vertices_m)
                expected = np.zeros(obj.mask.shape,np.uint8)
                # Compare the coloured top face, never the full side silhouette.
                top = cube.vertices_m[np.argsort(cube.vertices_m[:,2])[-4:]]
                cv2.fillConvexPoly(expected,cv2.convexHull(np.int32(self.geometry.project(top))),1)
                overlap = np.count_nonzero((expected>0)&(obj.mask>0))/max(1,np.count_nonzero((expected>0)|(obj.mask>0)))
                if overlap > .85:
                    obj.id = key
                    obj.pose[:3,:3],obj.pose[:3,3] = cube.rotation,cube.center_m
                    obj.dimensions = np.full(3,cube.side_m)
                    obj.evidence.update(cube.evidence,metric_qualified=True,metric_confirmation_iou=overlap)
        old = list({**{key:o for key,o in self.track_memory.items() if frame.timestamp-o.timestamp < 3.},**self.state.objects}.values())
        if old and objects:
            costs = np.array([[np.linalg.norm(a.pose[:3,3]-b.pose[:3,3]) + (0. if a.color == b.color else 10.) for b in objects] for a in old])
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row,column] < .04 and not objects[column].id:
                    objects[column].id = old[row].id
        for obj in objects:
            if self.held is not None and self.held[0] in self.state.objects and obj.color==self.state.objects[self.held[0]].color and sum(o.color==obj.color for o in objects)==1:
                obj.id,obj.status = self.held[0],'held'
            if self.pending_placement is not None and obj.color==self.pending_placement[1] and sum(o.color==obj.color for o in objects)==1:
                obj.id = self.pending_placement[0]
            if not obj.id:
                obj.id = f'cube-{self.next_id:03d}'
                self.next_id += 1
            if obj.id in self.reserved:
                obj.status = 'reserved'
        from so101_pick_place.geometry.cuboid_vision import cube_vertices
        for obj in objects:
            if obj.evidence.get('metric_qualified') and obj.status!='held':
                yaw = np.arctan2(obj.pose[1,0],obj.pose[0,0])
                vertices = cube_vertices(obj.pose[:2,3],obj.dimensions[0],yaw,obj.pose[2,3]-obj.dimensions[2]/2)
                self.metric_objects[obj.id] = SpatialCube(obj.pose[:3,3],obj.dimensions[0],obj.pose[:3,:3],vertices,dict(obj.evidence))
        self.track_memory.update({o.id:o for o in objects})
        self.track_memory = {key:o for key,o in self.track_memory.items() if frame.timestamp-o.timestamp < 3.}
        # Reuse box edges between periodic background measurements.
        if frame.timestamp != self.state.timestamp:
            proposals = self.track_box_edges(frame)
            if proposals is None:
                proposals = [] if self.boxes_semantically_seeded else detect_boxes(frame)
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
                measured = self.metric_boxes.get(box.id)
                if measured is not None:
                    previous = np.asarray(measured.evidence['polygon'])
                    if np.linalg.norm(previous.mean(0)-np.mean(proposal['polygon'],axis=0)) < 4.:
                        box.evidence.update(measured.evidence,polygon=proposal['polygon'])
                        box.pose = measured.pose.copy()
                        box.inner_dimensions = measured.inner_dimensions.copy()
                        box.wall_height = measured.wall_height
                boxes[box.id] = box
            boxes_at = frame.timestamp
        else:
            boxes, boxes_at = self.state.boxes, self.state.evidence['boxes_at']
        for box in boxes.values():
            polygon = np.array(box.evidence['polygon'],np.float32)
            box.occupied_slots = [o.id for o in objects if cv2.pointPolygonTest(polygon,tuple(np.mean(o.evidence['quad'],axis=0)),False) > 0]
        self.state = SceneState({o.id:o for o in objects},boxes,rejected,
            self.measured_surfaces or [{'id':'table','height_m':self.geometry.table_z,'source':'native calibrated plane'}],self.state.version+1,frame.timestamp,
            {'boxes_at':boxes_at,'coverage_verified':bool(self.metric_boxes) and len(boxes)==len(self.metric_boxes) and all(o.evidence.get('metric_qualified') for o in objects) and not rejected,'metric_camera':'overview','wrist_metric_qualified':False})
        self.raw = {'overview':detected,'wrist':wrist_detections}
        self.frames = dict(frames)
        if self.robot is not None and self.attachment_hypothesis is not None:
            try:
                self.passive_measurements = self.geometry.measure_visible_attachment(frame,self.robot,self.attachment_hypothesis)
            except ValueError as exc:
                self.passive_measurements = {'frame_id':frame.frame_id,'motion_authorized':False,'reason':str(exc)}
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
        self.pending_placement = None
        self.release_reservation(object_id)

    def find_free_box_slots(self, box_id, object_shape):
        box = self.state.boxes[box_id]
        if not box.evidence.get('metric_qualified'):
            raise ValueError('box opening, floor and wall height have no qualified metric survey')
        from box_extract.stow import box_floor_patches
        from box_extract.scene_builder import SceneParameters
        from types import SimpleNamespace
        cubes = {key:self.native_object(key) for key in self.state.objects if self.held is None or key!=self.held[0]}
        native = SimpleNamespace(box=box.evidence['native_box'],cubes=cubes)
        return box_floor_patches(native,SceneParameters())

    def find_free_table_slots(self, object_shape):
        from box_extract.stow import CUBE_KEEPOUT_LONG_M,CUBE_KEEPOUT_SHORT_M
        slots = []
        half = np.asarray(object_shape)[:2]/2
        finger_space = max(CUBE_KEEPOUT_LONG_M,CUBE_KEEPOUT_SHORT_M)
        for surface in self.state.surfaces:
            patch = surface.get('patch')
            if surface.get('free_surface_verified') and patch is not None:
                center = np.mean(patch.polygon_xy_m,axis=0)
                corners = center+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half
                blocked = any(np.all(np.abs(o.pose[:2,3]-center)<np.maximum(half+o.dimensions[:2]/2+.005,finger_space))
                              for o in self.state.objects.values() if o.status!='held')
                reserved = any(slot_id==surface['id'] for _,slot_id in self.reserved_slots)
                if not blocked and not reserved and patch.contains(np.column_stack((corners,np.full(4,patch.height_m)))):
                    slots.append(surface)
        return slots

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
        return SpatialCube(o.pose[:3,3],float(o.dimensions[0]),o.pose[:3,:3],vertices,o.evidence|{'object_id':object_id})

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
        if placement.support_object_id is not None:
            support = self.state.objects[placement.support_object_id]
            if self.build_support_graph()[support.id]:
                raise ValueError('support cube already carries another object')
            cube = self.native_object(support.id)
            patch = StackSupport(cube)
            primary = next((b.evidence.get('native_box') for b in self.state.boxes.values() if b.evidence.get('metric_qualified')),None)
            if primary is None:
                raise ValueError('visible boxes still need measured collision geometry')
        elif placement.support_surface in self.state.boxes:
            box = self.state.boxes[placement.support_surface]
            if not box.evidence.get('metric_qualified'):
                raise ValueError('box floor/rim/walls are not independently measured')
            patches = self.find_free_box_slots(box.id,self.state.objects[object_id].dimensions)
            if not patches:
                raise ValueError('no box slot remains free')
            patch = min(patches,key=lambda p:np.linalg.norm(np.mean(p.polygon_xy_m,axis=0)-placement.object_pose[:2,3]))
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
        obj = self.state.objects[object_id]
        expected_z = patch.height_m+obj.dimensions[2]/2
        if abs(placement.object_pose[2,3]-expected_z) > .004:
            raise ValueError('requested placement disagrees with measured support height')
        if placement.support_object_id is None:
            from so101_pick_place.experiments.box_extraction.scene import SupportPatch
            half = obj.dimensions[:2]/2+.007
            polygon = placement.object_pose[:2,3]+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half
            test = np.column_stack((polygon,np.full(4,patch.height_m)))
            if not patch.contains(test,margin_m=0.):
                raise ValueError('requested slot leaves the measured free support polygon')
            patch = SupportPatch(polygon,patch.height_m,patch.uncertainty_m,dict(patch.evidence))
        cubes = {key:self.native_object(key) for key in self.state.objects}
        boxes = [b.evidence['native_box'] for b in self.state.boxes.values()]
        return ExtractionScene(str(self.state.version),self.state.timestamp,BoxUnion(primary,boxes),cubes,patch,(),True)


    def remember_reset_layout(self):
        if not self.reset_targets:
            self.reset_targets = {o.id:Placement(o.pose.copy(),'table') for o in self.state.objects.values() if not any(o.id in b.occupied_slots for b in self.state.boxes.values())}
        return len(self.reset_targets)

    def install_metric_survey(self, surveys):
        """Accept actual native WristSurveyBuilder results; no nominal substitutes."""
        from scipy.optimize import linear_sum_assignment
        measured_objects = dict(self.metric_objects)
        for box_id,native in surveys.items():
            native.require_complete()
            box = self.state.boxes[box_id]
            box.evidence.update(metric_qualified=True,native_box=native.box)
            box.pose = np.asarray(native.box.base_from_box).copy()
            box.inner_dimensions = np.r_[native.box.inner_max_xy-native.box.inner_min_xy,native.box.obstacle_top_m-native.box.floor_interval_m[1]]
            box.wall_height = float(box.inner_dimensions[2])
            self.metric_boxes[box_id] = box
            current = list(self.state.objects.values());cubes = list(native.cubes.values())
            costs = np.array([[np.linalg.norm(o.pose[:2,3]-cube.center_m[:2]) for cube in cubes] for o in current])
            if current and cubes:
                rows,columns = linear_sum_assignment(costs)
                if len(columns)!=len(cubes) or any(costs[i,j]>.012 for i,j in zip(rows,columns)):
                    raise ValueError('wrist survey objects do not uniquely match current overview IDs')
                measured_objects.update({current[i].id:cubes[j] for i,j in zip(rows,columns)})
            self.measured_surfaces = [{'id':'table','height_m':native.placement.height_m,'patch':native.placement,'free_surface_verified':True}]
        self.metric_objects = measured_objects

    def install_measured_views(self, box_id, observations):
        """Reconstruct real floor/rim/wall tracks using the existing native builder."""
        from so101_pick_place.experiments.box_extraction.survey import WristSurveyBuilder,WristView
        views=[]
        for item in observations:
            frame=item['frame']
            if frame.joints is None or frame.gripper is None or frame.motion_during_exposure_rad is None or frame.motion_during_exposure_rad>.002:
                raise ValueError('box observations require stationary exposure-paired feedback')
            masks=[item[name] for name in ('box','opening','floor','free_table')]
            if any(np.asarray(mask).shape!=frame.image.shape[:2] for mask in masks):
                raise ValueError('box masks do not belong to their physical image')
            projection=self.geometry.projection_for_frame(frame,self.robot)
            views.append(WristView(frame.image,projection,frame.timestamp,self.robot.to_native(frame.joints,frame.gripper),
                tuple(item.get('cubes',())),*masks,tuple(item.get('unknown',())),frame.received_at))
        measured = WristSurveyBuilder().build(views)
        self.install_metric_survey({box_id:measured})
        return measured


class StackSupport:
    """Measured top face; stability uses the payload centre of mass."""
    def __init__(self, cube):
        from scipy.spatial import ConvexHull
        top = cube.vertices_m[np.argsort(cube.vertices_m[:,2])[-4:]]
        self.polygon_xy_m = top[ConvexHull(top[:,:2]).vertices,:2]
        self.height_m = float(top[:,2].mean())
        self.uncertainty_m = float(max(cube.evidence.get('position_sigma_m',[0.,0.,.002])))
        if self.uncertainty_m > .002 or np.ptp(top[:,2]) > .002:
            raise ValueError('tower support is too uncertain or tilted')
        self.support_id = cube.evidence.get('object_id')
        self.evidence = dict(cube.evidence)

    def contains(self, vertices, margin_m=.005):
        from scipy.spatial import ConvexHull
        hull = ConvexHull(self.polygon_xy_m)
        points = np.asarray(vertices)
        center = points.mean(0)[:2]
        radius = float(np.min(np.ptp(points[:,:2],axis=0)))/4+self.uncertainty_m
        return bool(np.all(center@hull.equations[:,:2].T+hull.equations[:,2] <= -radius))


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
