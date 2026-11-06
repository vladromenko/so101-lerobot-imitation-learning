"""Native mesh/IK/path checks and execution; no simulator and no fold-home."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from box_extract.driver import BoxArmDriver
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
        self.attachment_path = self.qualification_file.parent/'attachment-candidate.json'

    def qualification_reasons(self, geometry):
        reasons = []
        state = self.robot.last_state
        if state is None:
            reasons.append('нет свежей обратной связи приводов')
        else:
            audit = self.robot.audit_state(state)
            reasons.extend(audit['problems'])
            if audit['collision']:
                reasons.append(audit['collision'])
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
        for q in self.robot.native_edge_samples(start,goal):
            self.check_cancel()
            reason = self.state_is_valid(q,phase,plan,world)
            if reason:
                return reason
        return None

    def check_path(self, plan, world):
        for phase in plan.phases:
            self.report('checking',phase.name)
            for start,goal in zip(phase.joints,phase.joints[1:]):
                reason = self.edge_is_valid(start,goal,phase,plan,world)
                if reason:
                    raise ValueError(f'{phase.name}: {reason}')
                for q in (start,goal):
                    if self.planner.command_sag is None:
                        raise ValueError('native gravity-compensation setting has not been read')
                    reason = self.state_is_valid(self.planner._command_pose(q),phase,plan,world)
                    if reason:
                        raise ValueError(f'{phase.name} native command: {reason}')
        return True

    def plan_direct_transfer(self, start, goal, world, **policy):
        return self.planner.transport(start,goal,world,**policy)

    def execute_trajectory(self, phase, plan, world):
        self.check_cancel()
        if not self.robot.owns_lease() or self.robot.arm is None:
            raise RuntimeError('no exclusive native executor')
        if not self.contact_free_passed:
            raise RuntimeError('contact-free approach has not been physically qualified')
        self.report('moving',phase.name)
        self.driver.move_phase(phase,plan,world,self.check_cancel)

    def prepare_driver(self, watchdog):
        if self.robot.arm is None:
            raise RuntimeError('native driver must be connected once before Ready')
        self.driver = BoxArmDriver(self.robot.config,self.robot.backend,self.robot.output,
                                  arm=self.robot.arm,check_sensor=watchdog)
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
        evidence = {'passed':True,'timestamp':time.time(),'target_id':plan.target_id,
                    'start':start,'returned':actual,'tracking_error_rad':error,
                    'attachment_sha256':self.attachment.fingerprint,'contact_was_commanded':False}
        (self.qualification_file.parent/'contact-free-trial.json').write_text(json.dumps(evidence,indent=2))
        return evidence

    def return_to_start(self, start, plan, world):
        actual = self.robot.arm.read()[0]
        goal = dict(start,gripper=actual['gripper'])
        path,_ = self.planner.transport(actual,goal,world,target=plan.target_id,
                                       placed_cube=plan.phases[-1].placed_cube)
        phase = Phase('return_to_work',path,placed_cube=plan.phases[-1].placed_cube)
        self.check_path(replace(plan,phases=(phase,)),world)
        self.execute_trajectory(phase,plan,world)


class AccessoryWorld(ExtractionCollisionWorld):
    def __init__(self, robot, scene, attachment):
        super().__init__(robot.backend,scene)
        self.robot,self.attachment = robot,attachment

    def reason(self, joints, **policy):
        native = super().reason(joints,**policy)
        return native or self.attachment.reason(self.robot,joints,self.scene,policy.get('target'),policy.get('payload_local'))
