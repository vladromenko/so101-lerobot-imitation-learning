"""Native mesh/IK/path checks and execution; no simulator and no fold-home."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import cv2
from so101_pick_place.experiments.box_extraction.driver import ExistingArmDriver
from box_extract.planning import TunedExtractionPlanner
from so101_pick_place.experiments.box_extraction.planner import Phase, ExtractionPlan
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld
from .geometry import AttachmentGeometry


class Motion:
    def __init__(self, robot, check_cancel, report, qualification_file):
        self.robot, self.check_cancel, self.report = robot, check_cancel, report
        self.planner = TunedExtractionPlanner(robot.backend,check_cancel=check_cancel)
        self.metrics = {}
        self.planner.timings = self.metrics
        self.driver = None
        self.qualification_file = Path(qualification_file)
        self.world = self.plan = None
        self.contact_free_passed = False
        self.attachment = None
        self.trial_attachment_sha256 = None
        self.attachment_path = self.qualification_file.parent/'attachment-candidate.json'
        self.startup_world = None
        self.grasp_mode = 'wrist_refinement'
        self.operator_observation = None
        self.release_phase,self.release_previous = None,None

    def prepare_ready(self, scene, geometry, attendance, dry=False):
        """Leave the folded soft-margin violation through an attended opening corridor.

        This does not qualify an unknown cable or enable manipulation. The human
        observation covers the cable; the native model and rigid CAD are checked
        before and during the motion. Ordinary task margins remain unchanged.
        """
        from box_extract.mounted_arm import validated_segments
        self.check_cancel()
        if attendance.get('cable_and_hidden_surfaces_clear') is not True:
            raise ValueError('attended startup requires the operator observation')
        state = self.robot.read_state()
        if not self.robot.audit_state(state)['hard_valid']:
            raise ValueError('startup violates a hard constraint')
        start = state['positions']
        model = self.robot.backend.model
        ready = dict(start)
        # Only open the folded shoulder/elbow towards their ordinary domain.
        for index,name in ((1,'shoulder_lift'),(2,'elbow_flex')):
            low,high = model.ranges[index]
            ready[name] = float(np.clip(start[name],low+model.margin+1.5,high-model.margin-1.5))
        if max(abs(ready[j]-start[j]) for j in start)>10.:
            raise ValueError('initial opening exceeds the bounded ten-degree corridor')
        world = StartupWorld(self.robot,scene,geometry,start,ready,self.attachment_path)
        path = validated_segments(start,[ready],world.reason)
        sag = float(self.robot.arm.raw.sag)
        # Independent joint lag stays between the start and the waypoint. Check
        # its corners, rather than extrapolating past the stationary start.
        import itertools
        for q in path:
            self.check_cancel()
            lag_deg = self.planner.tracking_error_deg
            for lift,elbow in itertools.product((max(start['shoulder_lift'],q['shoulder_lift']-lag_deg),q['shoulder_lift']),
                                               (min(start['elbow_flex'],q['elbow_flex']+lag_deg),q['elbow_flex'])):
                lag = dict(q,shoulder_lift=lift,elbow_flex=elbow)
                for pose in (lag,self.robot.backend.driver_command_goal(lag,sag)):
                    reason = world.reason(pose)
                    if reason:
                        raise ValueError('startup path: '+reason)
        if model.problems(model.pose(ready)) or model.collides(model.pose(ready)):
            raise ValueError('startup endpoint must satisfy ordinary working constraints')
        evidence = {'start':start,'goal':ready,'attendance':attendance,'path':path,
                    'scope':'attended folded exit only','automatic_unfold':False,
                    'ordinary_margin_deg':model.margin,'ordinary_gap_m':model.gap,
                    'initial_sample_gap_m':world.initial_gap,'physical_completed':False}
        output = self.qualification_file.parent/'startup-trial.json'
        output.write_text(json.dumps(evidence,indent=2,ensure_ascii=False))
        if dry:
            return evidence
        self.startup_world = world
        self.report('moving','attended_prepare_ready: coordinated shoulder/elbow opening; wrist held')
        try:
            self.robot.enable()
            for name,goal in (('folded_exit',ready),):
                actual = self.robot.read_state()['positions']
                segment = validated_segments(actual,[goal],world.reason)
                for q in segment:
                    reason = world.reason(self.robot.backend.driver_command_goal(q,sag))
                    if reason:
                        raise ValueError('startup command: '+reason)
                travel = sum(max(abs(b[j]-a[j]) for j in a) for a,b in zip(segment,segment[1:]))
                self.robot.arm.follow(segment,duration=max(1.,1.875*travel/2.),settle=1.,within=.7,lead=False,step=name)
            final = self.robot.read_state()
            if not self.robot.audit_state(final)['planning_valid']:
                raise RuntimeError('startup ended outside ordinary working margins')
            evidence.update(physical_completed=True,actual=final['positions'],finished=time.time())
            output.write_text(json.dumps(evidence,indent=2,ensure_ascii=False))
            return evidence
        finally:
            self.startup_world = None
            self.robot.stop_motion()

    def qualification_reasons(self, geometry):
        return [item['reason'] for item in self.qualification_blockers(geometry)]

    def qualification_blockers(self, geometry):
        blockers = []
        def add(code,category,reason,condition):
            blockers.append({'code':code,'category':category,'reason':reason,'clear_when':condition})
        state = self.robot.last_state
        if state is None:
            add('FEEDBACK_MISSING','FEEDBACK','нет обратной связи приводов','эксклюзивный native-драйвер прочитал все энкодеры')
        else:
            if time.time()-state['timestamp']>.5:
                add('FEEDBACK_STALE','FEEDBACK','обратная связь приводов устарела','возраст измеренных углов не более 0,5 с')
            audit = self.robot.audit_state(state)
            if audit['hardware_limit_problems']:
                add('HARD_LIMIT','HARD_LIMIT','; '.join(audit['hardware_limit_problems']),'все энкодеры внутри сохранённых аппаратных диапазонов')
            if audit['hard_limit_problems'] and not audit['hardware_limit_problems']:
                add('KINEMATIC_CONSTRAINT','GEOMETRY','; '.join(audit['hard_limit_problems']),'выполнены native-ограничения геометрии без дополнительного запаса')
            if audit['zero_gap_sample_contact']:
                add('MODEL_CONTACT','GEOMETRY',audit['zero_gap_sample_contact'],'подтверждено отсутствие контакта всей руки; нулевой порог не обходить')
            if audit['hard_valid'] and not audit['planning_valid']:
                add('STARTUP_PREPARATION','STARTUP',
                    'штатная стартовая поза требует prepare_ready до применения рабочих запасов 5°/8 мм; это не отказ калибровки суставов',
                    'проверен начальный путь MountedTaskArm.prepare_ready из текущих энкодеров с камерой и отставанием приводов; рабочая конечная поза проходит обычные ограничения')
                blockers[-1]['working_margin_details'] = {'joints':audit['planning_margin_problems'],
                    'sample_separation_m':audit['minimum_model_clearance_m'],'pair':audit['planning_clearance_collision']}
        try:
            if self.grasp_mode=='overhead_grasp':
                self.attachment = AttachmentGeometry.attended(self.attachment_path,self.robot,self.operator_observation or {})
            else:
                self.attachment = AttachmentGeometry(self.attachment_path,self.qualification_file,geometry.fingerprint)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.attachment = None
            add('ACCESSORY_GEOMETRY','QUALIFICATION','нет квалифицированного объёма камеры/крепления/кабеля',
                'измеренная геометрия совпадает с калибровкой и независимо проверена для всего диапазона пути')
            blockers[-1]['evidence_error'] = str(exc)
        if geometry.wrist is None and self.grasp_mode!='overhead_grasp':
            add('WRIST_CALIBRATION','QUALIFICATION','нет независимо проверенной wrist hand-eye для текущего крепления',
                'установлена калибровка с независимой проверкой текущего крепления; отдельный флаг допуска наведения не требуется')
        return blockers

    def make_world(self, scene):
        if self.attachment is None:
            raise ValueError('accessory collision geometry is not qualified')
        return AccessoryWorld(self.robot,scene,self.attachment,self.metrics)

    def state_is_valid(self, q, phase, plan, world):
        return world.reason(q,target=plan.target_id,jaw_contact=phase.jaw_contact,
            payload_local=plan.payload_local if phase.carrying else None,support_contact=phase.support_contact,
            placed_cube=phase.placed_cube)

    def edge_is_valid(self, start, goal, phase, plan, world):
        if self.planner.command_sag is None:
            raise ValueError('native gravity-compensation setting has not been read')
        for q in self.robot.native_edge_samples(start,goal):
            self.check_cancel()
            for kind,pose in (('pose',q),('native command',self.planner._command_pose(q))):
                reason = self.state_is_valid(pose,phase,plan,world)
                intended = reason in {'gripper_link: cube '+plan.target_id,'moving_jaw_so101_v1_link: cube '+plan.target_id}
                force_close = phase.name=='grasp' and phase.jaw_contact and not phase.carrying
                stationary_arm = all(abs(goal[j]-start[j])<1e-6 for j in start if j!='gripper')
                if intended and force_close and stationary_arm and 4.<=pose['gripper']<=45.:
                    # Native grasp() closes by force, not by following these
                    # penetration requests as a rigid position trajectory.
                    reason = world.grasp_contact_reason(pose,plan.target_id)
                if reason=='gripper_link: cube '+plan.target_id and kind=='native command' and self.separating_release(goal,start,phase):
                    reason = world.grasp_contact_reason(pose,plan.target_id,phase.placed_cube)
                if reason:
                    return kind+': '+reason
        return None

    def check_path(self, plan, world):
        for phase in plan.phases:
            self.report('checking',phase.name)
            if not phase.joints:
                raise ValueError(phase.name+': empty trajectory')
            world.release_origin = phase.joints[0] if phase.name=='retract' and phase.placed_cube is not None else None
            try:
                edges = zip(phase.joints,phase.joints[1:]) if len(phase.joints)>1 else [(phase.joints[0],phase.joints[0])]
                for start,goal in edges:
                    reason = self.edge_is_valid(start,goal,phase,plan,world)
                    if reason:
                        raise ValueError(f'{phase.name}: {reason}')
                if phase.name=='retract':
                    reason = self.planner._tracking_robust(phase.joints,world,target=plan.target_id,
                        jaw_contact=phase.jaw_contact,placed_cube=phase.placed_cube)
                    if reason:
                        raise ValueError('retract: '+reason)
            finally:
                world.release_origin = None
        return True

    def plan_direct_transfer(self, start, goal, world, **policy):
        return self.planner.transport(start,goal,world,**policy)

    def informative_views(self, plan, world, geometry, point, current, radius_m=0.):
        """Choose viewpoints only on an already checked free-space approach."""
        from so101_pick_place.vision.wrist_camera_geometry import camera_projection
        if geometry.wrist is None or not self.contact_free_passed:
            raise ValueError('active perception requires calibrated wrist and a qualified free-space approach')
        projection = camera_projection(geometry.wrist,self.robot.backend.model,current)
        current_pose = np.linalg.inv(np.asarray(projection['camera_from_robot']))
        fixed = geometry.point_image_jacobian(point,geometry.calibration)
        size = np.asarray(geometry.wrist.get('image_size',[0,0]))
        if size.shape!=(2,) or np.any(size<=0):
            raise ValueError('active perception requires calibrated wrist image dimensions')
        candidates=[]
        for phase in plan.phases:
            if phase.name in ('approach','prepare_ready','observe') and not phase.carrying and not phase.jaw_contact:
                indices=np.unique(np.linspace(0,len(phase.joints)-1,min(12,len(phase.joints))).astype(int))
                for index in indices:
                    q=dict(phase.joints[index],gripper=current['gripper'])
                    observation=camera_projection(geometry.wrist,self.robot.backend.model,q)
                    pose=np.linalg.inv(np.asarray(observation['camera_from_robot']))
                    baseline=float(np.linalg.norm(pose[:3,3]-current_pose[:3,3]))
                    turn=float(np.arccos(np.clip((np.trace(pose[:3,:3].T@current_pose[:3,:3])-1)/2,-1.,1.)))
                    optical=np.asarray(observation['camera_from_robot'])@np.r_[point,1.]
                    k=np.asarray(observation['camera_matrix'])
                    camera=np.asarray(observation['camera_from_robot']);rotation,_=cv2.Rodrigues(camera[:3,:3])
                    pixel=cv2.projectPoints(np.asarray(point)[None],rotation,camera[:3,3],k,np.asarray(observation['distortion']))[0].ravel()
                    margin=max(k[0,0],k[1,1])*radius_m/max(.001,optical[2]-radius_m)+3*observation.get('reprojection_rmse_px',2.5)
                    in_view=bool(np.all(pixel>margin) and np.all(pixel<size-margin))
                    if optical[2]>.025+radius_m and in_view and (baseline>=.025 or turn>=np.deg2rad(5)) and world.reason(q,target=plan.target_id) is None:
                        moving=geometry.point_image_jacobian(point,observation)
                        information=fixed.T@fixed+moving.T@moving
                        score=float(np.linalg.eigvalsh(information)[0])
                        cost=float(np.linalg.norm(self.robot.from_native(q)-self.robot.from_native(current)))
                        candidates.append((-score,cost,q))
        if not candidates:
            raise ValueError('checked path offers no new informative view; movement is not bootstrapped from missing geometry')
        selected,poses=[],[]
        for _,_,q in sorted(candidates,key=lambda item:item[:2]):
            pose=np.linalg.inv(np.asarray(camera_projection(geometry.wrist,self.robot.backend.model,q)['camera_from_robot']))
            novel=all(np.linalg.norm(pose[:3,3]-old[:3,3])>=.025 or
                      np.arccos(np.clip((np.trace(pose[:3,:3].T@old[:3,:3])-1)/2,-1.,1.))>=np.deg2rad(5.) for old in poses)
            if novel:
                selected.append(q);poses.append(pose)
            if len(selected)>=3:
                break
        return selected

    def execute_trajectory(self, phase, plan, world):
        self.check_cancel()
        if not self.robot.owns_lease() or self.robot.arm is None:
            raise RuntimeError('no exclusive native executor')
        if not self.contact_free_passed or self.attachment is None or self.trial_attachment_sha256!=self.attachment.fingerprint:
            raise RuntimeError('contact-free approach has not been physically qualified')
        self.report('moving',phase.name)
        self._execute_native(phase,plan,world)

    def _execute_native(self, phase, plan, world):
        world.cache_enabled = False
        world.cache.clear()
        begin = time.perf_counter()
        try:
            world.release_origin = self.robot.arm.read()[0] if phase.name=='retract' and phase.placed_cube is not None else None
            self.driver.move_phase(phase,plan,world,self.check_cancel)
        except ValueError as exc:
            raise RuntimeError('native execution: '+str(exc)) from exc
        finally:
            world.release_origin = None
            self.metrics['actual_motion_s'] = (self.metrics.get('actual_motion_s',0.) or 0.)+time.perf_counter()-begin

    def prepare_driver(self, watchdog):
        if self.robot.arm is None:
            raise RuntimeError('native driver must be connected once before Ready')
        if self.driver is None or self.driver.arm is not self.robot.arm:
            self.driver = ExistingArmDriver(self.robot.config,self.robot.backend,self.robot.output,arm=self.robot.arm,check_sensor=watchdog)
            self.contact_free_passed = False
        self.driver.check_sensor = watchdog
        self.driver.mode = 'task'
        self.planner.command_sag = float(self.robot.arm.raw.sag)

    def guarded_tick(self, record):
        self.check_cancel()
        if self.startup_world is not None:
            for kind,q in (('goal',record['goal']),('pos',record['pos']),
                           ('command',self.robot.backend.driver_command_goal(record['goal'],self.robot.arm.raw.sag))):
                reason = self.startup_world.reason(q)
                if reason:
                    raise RuntimeError('startup '+kind+': '+reason)
            return
        if self.driver is None:
            raise RuntimeError('motion callback has no checked scene')
        phase = self.driver.phase
        previous = self.release_previous if self.release_phase is phase else (phase.joints[0] if phase is not None else record['pos'])
        try:
            self.driver.guard_tick(record)
        except RuntimeError as exc:
            phase,plan,world = self.driver.phase,self.driver.plan,self.driver.world
            expected = 'retract command: gripper_link: cube '+plan.target_id if plan is not None else ''
            if str(exc)!=expected or not self.separating_release(record['goal'],previous,phase):
                raise
            policy = {'target':plan.target_id,'jaw_contact':phase.jaw_contact,'placed_cube':phase.placed_cube}
            for kind,q in (('pos',record['pos']),('goal',record['goal'])):
                reason = world.reason(q,**policy)
                if reason:
                    raise RuntimeError('release separation '+kind+': '+reason) from exc
            command = self.robot.backend.driver_command_goal(record['goal'],self.robot.arm.raw.sag)
            reason = world.grasp_contact_reason(command,plan.target_id,phase.placed_cube)
            if reason:
                raise RuntimeError('release separation command: '+reason) from exc
            self.driver.previous['command'] = command
        self.release_phase,self.release_previous = phase,dict(record['goal'])

    def separating_release(self,goal,previous,phase):
        if phase is None or phase.name!='retract' or phase.placed_cube is None or phase.carrying:
            return False
        from box_extract.driver import BoxArmDriver
        start = phase.joints[0]
        model = self.robot.backend.model
        distance = np.linalg.norm(np.asarray(model.pose(goal).tip)-model.pose(start).tip)
        first_gap = self.planner._fixed_jaw_clearance(start,phase.placed_cube)
        before = self.planner._fixed_jaw_clearance(previous,phase.placed_cube)
        after = self.planner._fixed_jaw_clearance(goal,phase.placed_cube)
        return bool(distance<=BoxArmDriver.SEPARATION_ZONE_M and after>=max(first_gap,before)-1e-6 and
                    abs(goal['gripper']-start['gripper'])<1e-6)

    def contact_free_trial(self, plan, world, start):
        """Qualify the empty-hand approach, ending at its checked pregrasp."""
        outward = tuple(p for p in plan.phases if p.name in ('open','approach'))
        if not outward or any(p.carrying or p.jaw_contact for p in outward):
            raise ValueError('contact-free trial must contain only clear approach phases')
        self.check_path(replace(plan,phases=outward),world)
        world.cache_enabled = False
        world.cache.clear()
        self.driver.phase,self.driver.plan,self.driver.world = outward[0],plan,world
        self.robot.enable()
        for phase in outward:
            self.report('contact_free',phase.name)
            self._execute_native(phase,plan,world)
        actual = self.robot.arm.read()[0]
        reason = world.reason(actual,target=plan.target_id)
        if reason:
            raise RuntimeError('noncontact pregrasp endpoint: '+reason)
        self.contact_free_passed = True
        self.trial_attachment_sha256 = self.attachment.fingerprint
        evidence = {'passed':True,'timestamp':time.time(),'target_id':plan.target_id,
                    'start':start,'pregrasp_measured':actual,'ended_at':'pregrasp',
                    'attachment_sha256':self.attachment.fingerprint,'contact_was_commanded':False}
        (self.qualification_file.parent/'contact-free-trial.json').write_text(json.dumps(evidence,indent=2))
        return evidence

    def replan_held(self, scene, current, previous, original):
        """Only the force-confirmed payload is removed from the static map."""
        from so101_pick_place.geometry.spatial_cube import SpatialCube
        target,local = previous.target_id,previous.payload_local
        if target in scene.cubes:
            raise ValueError('attached cube remains in static obstacle map')
        frame = self.robot.backend.model.frames(current)['gripper_frame_link']
        vertices = self.robot.backend._transform_points(frame,local)
        rotation_local = np.linalg.lstsq(original.vertices_m-original.center_m,local-local.mean(0),rcond=None)[0].T @ original.rotation
        cube = SpatialCube(vertices.mean(0),original.side_m,frame[:3,:3]@rotation_local,vertices,original.evidence)
        updated = replace(scene,cubes=scene.cubes|{target:cube})
        world = self.make_world(updated)
        policy = {'target':target,'jaw_contact':True,'payload_local':local}
        rim = float((scene.box.base_from_box@np.r_[0.,0.,scene.box.obstacle_top_m,1.])[2])
        safe_bottom = max(rim+self.planner.rim_margin_m, float(scene.placement.height_m)+.025)
        rise = max(.025,safe_bottom-float(vertices[:,2].min()))
        try:
            lift = self.planner.vertical_center(current,rise,local.mean(0),world,support_contact=True,**policy)
        except ValueError as exc:
            raise ValueError('extract_vertical: '+str(exc)) from exc
        patch = scene.placement
        center = np.r_[np.mean(patch.polygon_xy_m,axis=0),patch.height_m+patch.uncertainty_m+cube.side_m/2+.001]
        def admissible(q):
            return all(world.reason(pose,support_contact=True,**policy) is None
                       for pose in (q,self.planner._command_pose(q)))
        down = self.planner.placement_pose(current,local,cube,center,accept_pose=admissible)
        placed_frame = self.robot.backend.model.frames(down)['gripper_frame_link']
        placed_vertices = self.robot.backend._transform_points(placed_frame,local)
        if not patch.contains(placed_vertices):
            raise ValueError('support cannot stably contain the held payload')
        rise_at_place = max(.025,safe_bottom-float(placed_vertices[:,2].min()))
        try:
            up = self.planner.vertical_center(down,rise_at_place,local.mean(0),world,support_contact=True,**policy)
        except ValueError as exc:
            raise ValueError('lower_to_support: '+str(exc)) from exc
        try:
            transfer,search = self.planner.transport(lift[-1],up[-1],world,**policy)
        except ValueError as exc:
            raise ValueError('transfer: '+str(exc)) from exc
        opening = next(p.joints[-1]['gripper'] for p in previous.phases if p.name=='release')
        goal,release = self.planner._release(down,world,first=opening,support_contact=True,**policy)
        placed = replace(cube,center_m=placed_vertices.mean(0),vertices_m=placed_vertices,rotation=placed_frame[:3,:3]@rotation_local)
        retreat = self.planner._retract_path(goal,local.mean(0),world,target=target,jaw_contact=True,placed_cube=placed)
        # Keep the native support-contact policy throughout this upward escape.
        # Splitting at the first nominal millimetre of clearance switches policy
        # before geared servos have actually left the support. The native guard
        # still forbids penetration below support - 1 mm and checks every wall.
        phases = (Phase('extract_vertical',lift,True,True,True),Phase('transfer',transfer,True,True),
                  Phase('lower_to_support',up[::-1],True,True,True),Phase('release',release,True,True,True),
                  Phase('retract',retreat,jaw_contact=True,placed_cube=placed))
        return ExtractionPlan(scene.revision,target,phases,local,center,(search,)),updated

    def return_to_start(self, start, plan, world, observed_cube):
        actual = self.robot.arm.read()[0]
        goal = dict(start,gripper=actual['gripper'])
        path,_ = self.planner.transport(actual,goal,world,target=plan.target_id,
                                       placed_cube=observed_cube)
        phase = Phase('return_to_work',path,placed_cube=observed_cube)
        self.check_path(replace(plan,phases=(phase,)),world)
        self.execute_trajectory(phase,plan,world)

    def recover_empty_approach(self,plan,world):
        """Return on the already checked, empty-hand approach after a STOP."""
        from dataclasses import replace
        self.check_cancel()
        self.prepare_driver(self.driver.check_sensor)
        current = self.robot.read_state()['positions']
        approach = next(p for p in plan.phases if p.name=='approach')
        nearest = min(range(len(approach.joints)),key=lambda i:max(abs(current[j]-approach.joints[i][j]) for j in current if j!='gripper'))
        reverse = tuple(dict(q,gripper=current['gripper']) for q in reversed(approach.joints[:nearest+1]))
        bridge = self.planner.edge(current,reverse[0],world)
        back = Phase('recover_empty_approach',tuple(bridge)+reverse[1:])
        recovery = replace(plan,phases=(back,))
        self.check_path(recovery,world)
        self.driver.phase,self.driver.plan,self.driver.world = back,recovery,world
        self.robot.enable()
        for phase in recovery.phases:
            self.report('moving',phase.name)
            self._execute_native(phase,recovery,world)
        actual = self.robot.read_state()['positions']
        start = plan.phases[0].joints[0]
        restore = Phase('open',self.planner.edge(actual,dict(start,gripper=start['gripper']),world))
        self.check_path(replace(plan,phases=(restore,)),world)
        self._execute_native(restore,plan,world)
        returned = self.robot.read_state()['positions']
        error = float(np.max(np.abs(self.robot.from_native(returned)-self.robot.from_native(start))))
        if error>np.pi/240.:
            raise RuntimeError('resumed noncontact trial did not return within tracking tolerance')
        evidence = {'returned_safely':True,'contact_free_trial_complete':False,'start':start,'returned':returned,
                    'tracking_error_rad':error,'attachment_sha256':self.attachment.fingerprint,'contact_was_commanded':False}
        (self.qualification_file.parent/'empty-approach-return.json').write_text(json.dumps(evidence,indent=2))
        return returned


class StartupWorld:
    """Checked, pose-bounded exit; never used as a manipulation collision world."""
    def __init__(self,robot,scene,geometry,start,goal,profile_path):
        from scipy.spatial import ConvexHull
        from .diagnostics import minimum_clearance
        from so101_pick_place.experiments.box_extraction.mesh_collision import load_collision_meshes
        self.robot,self.geometry = robot,geometry
        self.initial_gap = minimum_clearance(robot.backend.model,start)
        self.start,self.goal = dict(start),dict(goal)
        self.meshes = load_collision_meshes(str(robot.root/'vendor/SO101/so101_new_calib.urdf'))
        self.points = {}
        for link,parts in self.meshes.items():
            points = np.unique(np.concatenate([part.triangles.reshape(-1,3) for part in parts]),axis=0)
            self.points[link] = points[ConvexHull(points).vertices]
        profile = json.loads(Path(profile_path).read_text())
        self.parts = profile['parts']
        self.mask = np.zeros(scene.frames['overview'].image.shape[:2],np.uint8)
        for box in scene.state.boxes.values():
            cv2.fillConvexPoly(self.mask,np.int32(box.evidence['polygon']),255)
        for obj in scene.state.objects.values():
            self.mask[obj.mask>0] = 255
        self.mask = cv2.dilate(self.mask,np.ones((25,25),np.uint8))
        self.cache = {}

    def reason(self,q,**policy):
        key = tuple(q[j] for j in self.start)
        if key not in self.cache:
            if len(self.cache)>8192:
                self.cache.clear()
            self.cache[key] = self._reason(q)
        return self.cache[key]

    def _reason(self,q):
        from so101_pick_place.experiments.box_extraction.mesh_collision import mesh_box_overlap
        robot,model = self.robot,self.robot.backend.model
        for name,value in q.items():
            low,high = sorted((self.start[name],self.goal[name]))
            # Native sag can move the elbow inwards; no outward range override.
            allowance = 2. if name=='elbow_flex' else .8
            if not low-allowance<=value<=high+.8:
                return name+': outside bounded startup corridor'
        pose = model.pose(q)
        problems = model.problems(pose,margin=0.)
        if problems:
            return 'hard constraints: '+str(problems)
        collision = model.collides(pose,gap=max(0.,self.initial_gap-.0002))
        if collision:
            return 'startup closes existing sampled separation: '+collision
        frames = model.frames(q)
        bodies = dict(self.points)
        for part in self.parts:
            lo,hi = np.asarray(part['lower_m']),np.asarray(part['upper_m'])
            local = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
            parent = frames[part['parent_link']]
            inverse = np.linalg.inv(parent)
            for link,points in self.points.items():
                other = robot.backend._transform_points(inverse@frames[link],points)
                separated = np.any(other.max(0)<lo) or np.any(other.min(0)>hi)
                if link not in part.get('allow_contact_links',[]) and not separated:
                    if mesh_box_overlap(self.meshes[link],frames[link],parent,lo,hi):
                        return part['name']+': robot link '+link
            bodies[part['name']] = local
            frames[part['name']] = parent
        for name,local in bodies.items():
            points = robot.backend._transform_points(frames[name],local)
            if name not in ('base_link','shoulder_link') and points[:,2].min()<model.table_z+.004:
                return name+': table clearance'
            hull = cv2.convexHull(np.float32(self.geometry.project(points))).astype(np.int32)
            mask = np.zeros_like(self.mask)
            cv2.fillConvexPoly(mask,hull,255)
            if np.any((mask>0)&(self.mask>0)):
                return name+': projected cube/box keepout'
        return None


class AccessoryWorld(ExtractionCollisionWorld):
    def __init__(self, robot, scene, attachment, metrics=None):
        from types import SimpleNamespace
        super().__init__(robot.backend,scene)
        self.robot,self.attachment = robot,attachment
        self.metrics = {} if metrics is None else metrics
        self.cache,self.cache_enabled = {},True
        # The general payload floor is the calibrated table. A destination's
        # elevated support applies only at placement, never at the source pick.
        ground = SimpleNamespace(height_m=float(robot.backend.model.table_z),uncertainty_m=0.)
        self.ground_scene = replace(scene,placement=ground)
        self.native_checks = ExtractionCollisionWorld(robot.backend,self.ground_scene)
        self.grasp_worlds = {}
        self.release_origin = None

    def grasp_contact_reason(self,joints,target,placed_cube=None):
        """Permit only target/jaw contact; independently check everything else."""
        from so101_pick_place.experiments.box_extraction.mesh_collision import mesh_cube_overlap
        from so101_pick_place.experiments.box_extraction.scene import localization_margin
        if target not in self.grasp_worlds:
            other = replace(self.ground_scene,cubes={key:cube for key,cube in self.ground_scene.cubes.items() if key!=target})
            self.grasp_worlds[target] = ExtractionCollisionWorld(self.robot.backend,other)
        reason = self.grasp_worlds[target].reason(joints)
        if reason:
            return reason
        cube = self.scene.cubes[target] if placed_cube is None else placed_cube
        frames = self.robot.backend.model.frames(joints)
        for link,parts in self.collision_meshes.items():
            if link not in ('gripper_link','moving_jaw_so101_v1_link'):
                if mesh_cube_overlap(parts,frames[link],cube,localization_margin(cube)):
                    return link+': cube '+target
        return self.attachment.reason(self.robot,joints,self.scene,target,placed_cube=placed_cube)

    def support_contact_valid(self, joints, payload, support_id):
        support = self.scene.cubes[support_id]
        frame = self.robot.backend.model.frames(joints)['gripper_frame_link']
        points = self.robot.backend._transform_points(frame,payload)
        top = support.vertices_m[np.argsort(support.vertices_m[:,2])[-4:]]
        if np.ptp(top[:,2]) > .002 or points[:,2].min() < top[:,2].mean()-.001:
            return False
        from scipy.spatial import ConvexHull
        hull = ConvexHull(top[:,:2])
        center = points.mean(0)[:2]
        margin = float(np.min(np.ptp(points[:,:2],axis=0)))/4
        return bool(np.all(center@hull.equations[:,:2].T+hull.equations[:,2] <= -margin))

    def reason(self, joints, **policy):
        payload,placed = policy.get('payload_local'),policy.get('placed_cube')
        key = None
        if self.cache_enabled:
            key = (tuple(sorted(joints.items())),policy.get('target'),policy.get('jaw_contact',False),
                   policy.get('support_contact',False),None if payload is None else np.asarray(payload).tobytes(),
                   None if placed is None else (float(placed.side_m),placed.vertices_m.tobytes(),placed.rotation.tobytes()),
                   None if self.release_origin is None else tuple(sorted(self.release_origin.items())))
            if key in self.cache:
                self.metrics['collision_cache_hits'] = self.metrics.get('collision_cache_hits',0)+1
                return self.cache[key]
        begin = time.perf_counter()
        try:
            result = self._reason(joints,**policy)
        finally:
            self.metrics['collision_checking_s'] = (self.metrics.get('collision_checking_s',0.) or 0.)+time.perf_counter()-begin
            self.metrics['collision_checks'] = self.metrics.get('collision_checks',0)+1
        if key is not None:
            if len(self.cache)>=8192:
                self.cache.clear()
            self.cache[key] = result
        return result

    def _reason(self, joints, **policy):
        native = self.native_checks.reason(joints,**policy)
        target = policy.get('target')
        placed = policy.get('placed_cube')
        if native=='gripper_link: cube '+str(target) and placed is not None and self.release_origin is not None and policy.get('jaw_contact') and policy.get('payload_local') is None:
            from box_extract.driver import BoxArmDriver
            origin = self.release_origin
            model = self.robot.backend.model
            distance = np.linalg.norm(np.asarray(model.pose(joints).tip)-model.pose(origin).tip)
            def gap(q):
                frame = model.frames(q)['gripper_frame_link']
                return float(-((placed.vertices_m-frame[:3,3])@frame[:3,:3])[:,0].max())
            if distance<=BoxArmDriver.SEPARATION_ZONE_M and gap(joints)>=gap(origin)-1e-6 and joints['gripper']>=origin['gripper']-.7:
                native = self.grasp_contact_reason(joints,target,placed)
        jaw_contact = native in {'gripper_link: cube '+str(target),'moving_jaw_so101_v1_link: cube '+str(target)}
        if jaw_contact and policy.get('jaw_contact') and policy.get('payload_local') is None:
            other = self.grasp_contact_reason(joints,target,policy.get('placed_cube'))
            if other:
                return other
        support_id = getattr(self.scene.placement,'support_id',None)
        intended = support_id is not None and native == 'payload: cube '+support_id
        if intended and policy.get('support_contact') and policy.get('payload_local') is not None:
            if self.support_contact_valid(joints,policy['payload_local'],support_id):
                remaining = replace(self.ground_scene,cubes={key:value for key,value in self.scene.cubes.items() if key!=support_id})
                native = ExtractionCollisionWorld(self.robot.backend,remaining).reason(joints,**policy)
        return native or self.attachment.reason(self.robot,joints,self.scene,policy.get('target'),policy.get('payload_local'),policy.get('placed_cube'))
