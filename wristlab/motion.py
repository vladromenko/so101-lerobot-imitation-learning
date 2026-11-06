"""Native mesh/IK/path checks and execution; no simulator and no fold-home."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from so101_pick_place.experiments.box_extraction.driver import ExistingArmDriver
from box_extract.planning import TunedExtractionPlanner
from so101_pick_place.experiments.box_extraction.planner import Phase, ExtractionPlan
from so101_pick_place.experiments.box_extraction.scene import ExtractionCollisionWorld
from .geometry import AttachmentGeometry


class Motion:
    def __init__(self, robot, check_cancel, report, qualification_file):
        self.robot, self.check_cancel, self.report = robot, check_cancel, report
        self.planner = TunedExtractionPlanner(robot.backend,check_cancel=check_cancel)
        self.driver = None
        self.qualification_file = Path(qualification_file)
        self.world = self.plan = None
        self.contact_free_passed = False
        self.attachment = None
        self.trial_attachment_sha256 = None
        self.attachment_path = self.qualification_file.parent/'attachment-candidate.json'

    def qualification_reasons(self, geometry):
        reasons = []
        state = self.robot.last_state
        if state is None:
            reasons.append('нет свежей обратной связи приводов')
        else:
            if time.time()-state['timestamp']>.5:
                reasons.append('обратная связь приводов устарела')
            audit = self.robot.audit_state(state)
            if audit['hard_valid']:
                reasons.extend('не выдержан запас '+str(round(np.rad2deg(audit['required_joint_margin_rad'])))+'°: '+reason for reason in audit['planning_margin_problems'])
                if audit['planning_clearance_collision']:
                    reasons.append('зазор '+str(round(1000*audit['minimum_model_clearance_m'],1))+' мм меньше штатных '+str(round(1000*audit['required_clearance_m']))+' мм: '+audit['planning_clearance_collision'])
            else:
                reasons.extend(audit['hard_limit_problems'])
                if audit['actual_mesh_collision']:
                    reasons.append(audit['actual_mesh_collision'])
        try:
            evidence = json.loads(self.qualification_file.read_text())
        except (OSError, ValueError):
            evidence = {}
        if evidence.get('overview_calibration_sha256') != geometry.fingerprint:
            reasons.append('не подтверждена независимая проверка текущей геометрии камеры/крепления/кабеля')
        for key, text in (('accessory_geometry_verified','геометрия крепления и камеры не квалифицирована'),
                          ('cable_sweep_verified','объём и движение кабеля не квалифицированы'),
                          ('wrist_alignment_verified','wrist-наведение для текущего крепления не квалифицировано')):
            if evidence.get(key) is not True:
                reasons.append(text)
        try:
            self.attachment = AttachmentGeometry(self.attachment_path,self.qualification_file,geometry.fingerprint)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            reasons.append('нет проверенной геометрии камеры/кабеля: '+str(exc))
        if geometry.wrist is None:
            reasons.append('нет wrist hand-eye, прошедшей независимую проверку текущего крепления')
        return reasons

    def make_world(self, scene):
        if self.attachment is None:
            raise ValueError('accessory collision geometry is not qualified')
        return AccessoryWorld(self.robot,scene,self.attachment)

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
                if reason:
                    return kind+': '+reason
        return None

    def check_path(self, plan, world):
        for phase in plan.phases:
            self.report('checking',phase.name)
            if not phase.joints:
                raise ValueError(phase.name+': empty trajectory')
            edges = zip(phase.joints,phase.joints[1:]) if len(phase.joints)>1 else [(phase.joints[0],phase.joints[0])]
            for start,goal in edges:
                reason = self.edge_is_valid(start,goal,phase,plan,world)
                if reason:
                    raise ValueError(f'{phase.name}: {reason}')
        return True

    def plan_direct_transfer(self, start, goal, world, **policy):
        return self.planner.transport(start,goal,world,**policy)

    def execute_trajectory(self, phase, plan, world):
        self.check_cancel()
        if not self.robot.owns_lease() or self.robot.arm is None:
            raise RuntimeError('no exclusive native executor')
        if not self.contact_free_passed or self.attachment is None or self.trial_attachment_sha256!=self.attachment.fingerprint:
            raise RuntimeError('contact-free approach has not been physically qualified')
        self.report('moving',phase.name)
        self.driver.move_phase(phase,plan,world,self.check_cancel)

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
        if self.driver is None:
            raise RuntimeError('motion callback has no checked scene')
        self.driver.guard_tick(record)

    def contact_free_trial(self, plan, world, start):
        """Physically exercise only the free-space approach and its checked reverse."""
        outward = tuple(p for p in plan.phases if p.name in ('open','approach'))
        if not outward or any(p.carrying or p.jaw_contact for p in outward):
            raise ValueError('contact-free trial must contain only clear approach phases')
        self.check_path(replace(plan,phases=outward),world)
        self.driver.phase,self.driver.plan,self.driver.world = outward[0],plan,world
        self.robot.enable()
        for phase in outward:
            self.report('contact_free',phase.name)
            self.driver.move_phase(phase,plan,world,self.check_cancel)
        current = self.robot.arm.read()[0]
        reverse = tuple(reversed(outward[-1].joints))
        join = self.planner.edge(current,reverse[0],world,target=plan.target_id)
        back = Phase('contact_free_return',tuple(join)+reverse[1:])
        self.check_path(replace(plan,phases=(back,)),world)
        self.driver.move_phase(back,plan,world,self.check_cancel)
        current = self.robot.arm.read()[0]
        restore = Phase('open',self.planner.edge(current,start,world,target=plan.target_id))
        self.check_path(replace(plan,phases=(restore,)),world)
        self.driver.move_phase(restore,plan,world,self.check_cancel)
        actual = self.robot.arm.read()[0]
        error = float(np.max(np.abs(self.robot.from_native(actual)-self.robot.from_native(start))))
        if error > np.pi/240.:
            raise RuntimeError('contact-free trial did not return within tracking tolerance')
        self.contact_free_passed = True
        self.trial_attachment_sha256 = self.attachment.fingerprint
        evidence = {'passed':True,'timestamp':time.time(),'target_id':plan.target_id,
                    'start':start,'returned':actual,'tracking_error_rad':error,
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
        lift = self.planner.vertical_center(current,rise,local.mean(0),world,**policy)
        patch = scene.placement
        center = np.r_[np.mean(patch.polygon_xy_m,axis=0),patch.height_m+patch.uncertainty_m+cube.side_m/2+.001]
        down = self.planner.placement_pose(current,local,cube,center)
        placed_frame = self.robot.backend.model.frames(down)['gripper_frame_link']
        placed_vertices = self.robot.backend._transform_points(placed_frame,local)
        if not patch.contains(placed_vertices):
            raise ValueError('support cannot stably contain the held payload')
        rise_at_place = max(.025,safe_bottom-float(placed_vertices[:,2].min()))
        up = self.planner.vertical_center(down,rise_at_place,local.mean(0),world,support_contact=True,**policy)
        transfer,search = self.planner.transport(lift[-1],up[-1],world,**policy)
        opening = next(p.joints[-1]['gripper'] for p in previous.phases if p.name=='release')
        goal,release = self.planner._release(down,world,first=opening,support_contact=True,**policy)
        placed = replace(cube,center_m=placed_vertices.mean(0),vertices_m=placed_vertices,rotation=placed_frame[:3,:3]@rotation_local)
        retreat = self.planner._retract_path(goal,local.mean(0),world,target=target,jaw_contact=True,placed_cube=placed)
        phases = (Phase('extract_vertical',lift,True,True),Phase('transfer',transfer,True,True),
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


class AccessoryWorld(ExtractionCollisionWorld):
    def __init__(self, robot, scene, attachment):
        from types import SimpleNamespace
        super().__init__(robot.backend,scene)
        self.robot,self.attachment = robot,attachment
        # The general payload floor is the calibrated table. A destination's
        # elevated support applies only at placement, never at the source pick.
        ground = SimpleNamespace(height_m=float(robot.backend.model.table_z),uncertainty_m=0.)
        self.ground_scene = replace(scene,placement=ground)
        self.native_checks = ExtractionCollisionWorld(robot.backend,self.ground_scene)

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
        native = self.native_checks.reason(joints,**policy)
        support_id = getattr(self.scene.placement,'support_id',None)
        intended = support_id is not None and native == 'payload: cube '+support_id
        if intended and policy.get('support_contact') and policy.get('payload_local') is not None:
            if self.support_contact_valid(joints,policy['payload_local'],support_id):
                remaining = replace(self.ground_scene,cubes={key:value for key,value in self.scene.cubes.items() if key!=support_id})
                native = ExtractionCollisionWorld(self.robot.backend,remaining).reason(joints,**policy)
        return native or self.attachment.reason(self.robot,joints,self.scene,policy.get('target'),policy.get('payload_local'),policy.get('placed_cube'))
