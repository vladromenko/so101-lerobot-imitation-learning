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


class NativeRecoveryMargins:
    """Only the initially deficient native pairs receive their measured bound."""
    def __init__(self, model, minimums):
        self.model,self.minimums = model,minimums

    def __getattr__(self, name):
        return getattr(self.model,name)

    def collides(self, pose, gap=None):
        if gap is not None and gap!=self.model.gap:
            return self.model.collides(pose,gap=gap)
        hits = self.model.pair_collisions(pose,self.minimums)
        return hits[0] if hits else None


class RecoveryCollisionWorld(ExtractionCollisionWorld):
    """The native checker, with only measured initial link/wall gaps scoped."""
    def __init__(self, backend, scene, minimums, physical):
        super().__init__(backend,scene)
        self.minimums,self.physical = minimums,physical

    def remaining_wall(self, name, frame):
        from so101_pick_place.experiments.box_extraction.mesh_collision import mesh_box_overlap
        parts = self.collision_meshes[name]
        vertices = np.concatenate([p.triangles.reshape(-1,3) for p in parts])
        points = self.backend._transform_points(frame,vertices)
        inverse = np.linalg.inv(self.scene.box.base_from_box)
        local = points@inverse[:3,:3].T+inverse[:3,3]
        bounds_lo,bounds_hi = local.min(0),local.max(0)
        for lower,upper,label in self.scene.box.solid_parts():
            if np.all(bounds_hi>=lower) and np.all(bounds_lo<=upper):
                minimum = self.minimums.get((name,label))
                gap = None
                if minimum is not None:
                    lo,hi = self.physical[label]
                    gap = float(np.max(np.maximum(lo-bounds_hi,bounds_lo-hi)))
                # A separating full-mesh bound proves every triangle clear.
                # Otherwise retain the exact native triangle/box intersection.
                if (minimum is None or gap<minimum) and mesh_box_overlap(parts,frame,self.scene.box.base_from_box,lower,upper):
                    return label if minimum is None else 'recovery decreases '+label
        return None


def _native_recovery_reason():
    # Like the existing scaled_move adapter: retain the native method and
    # insert one phase-specific hook; do not fork the collision algorithm.
    import inspect,sys,textwrap
    source = textwrap.dedent(inspect.getsource(ExtractionCollisionWorld.reason))
    original = "if wall is not None:\n                return f'{name}: {wall}'"
    replacement = "if wall is not None:\n                wall = self.remaining_wall(name,frames[name])\n                if wall is not None:\n                    return f'{name}: {wall}'"
    if source.count(original)!=1:
        raise RuntimeError('native collision method changed; recovery hook refused')
    namespace = dict(vars(sys.modules[ExtractionCollisionWorld.__module__]))
    exec(compile(source.replace(original,replacement),'<native phase recovery>','exec'),namespace)
    return namespace['reason']


RecoveryCollisionWorld.reason = _native_recovery_reason()


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
        self.recovery_report = None
        self.recovery_mapping = None
        self.executing_recovery = False

    def prepare_ready(self, scene, geometry, attendance, dry=False, planning_check=None):
        """Leave the folded soft-margin violation through an attended opening corridor.

        This does not qualify an unknown cable or enable manipulation. The human
        observation covers the cable; the native model and rigid CAD are checked
        before and during the motion. Ordinary task margins remain unchanged.
        """
        from box_extract.mounted_arm import validated_segments
        check = planning_check if dry and planning_check is not None else self.check_cancel
        check()
        if not dry and attendance.get('cable_and_hidden_surfaces_clear') is not True:
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
            check()
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
        if dry:
            return evidence
        output.write_text(json.dumps(evidence,indent=2,ensure_ascii=False))
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

    def overview_clear_path(self, scene, geometry, check):
        """Select the shortest checked base rotation that exposes an occluded cube."""
        from box_extract.scene_builder import robot_mask
        state = self.robot.read_state()
        start = state['positions']
        if not self.robot.audit_state(state)['planning_valid'] or not self.robot.arm.torque:
            raise ValueError('a powered ordinary working pose is required for a new view')
        if state['loads']['gripper'] or state['currents']['gripper']:
            raise ValueError('view change requires an empty gripper')
        mapping = self.robot.synchronize_reference(state,self.robot.arm.raw.sag)
        if not mapping['supported_stop_packet_checked']:
            raise ValueError('preservation of the last checked holding goals was not confirmed')
        model = self.robot.backend.model
        frame = scene.frames['overview']
        world = StartupWorld(self.robot,scene,geometry,start,start,self.attachment_path,strict=True)
        if not world.partial_cubes:
            raise ValueError('no bounded partially visible cube requires a new overview')
        visible = np.zeros(frame.image.shape[:2],np.uint8)
        for _,_,_,rectangle in world.partial_cubes:
            cv2.fillConvexPoly(visible,rectangle,255)
        limits = model.ranges[0][0]+model.margin,model.ranges[0][1]-model.margin
        pan = 'shoulder_pan'
        candidates = sorted((q for limit in limits for q in self.robot.native_edge_samples(start,dict(start,**{pan:limit}))
                             if q[pan]!=start[pan]),key=lambda q:abs(q[pan]-start[pan]))
        first_refusal = None
        rejected = set()
        for goal in candidates:
            check()
            # Require the whole visible fragment, not just its centre, to be
            # clear in the prospective image. The new frame must then localize it.
            side = np.sign(goal[pan]-start[pan])
            occlusion = robot_mask(self.robot.backend,goal,geometry.calibration,visible.shape) if side not in rejected else visible
            if not np.any((occlusion>0)&(visible>0)):
                world.goal = goal
                world.cache.clear()
                world.partial_clearance = {}
                path = tuple(self.robot.native_edge_samples(start,goal))
                self.recovery_mapping = mapping
                holding = mapping['hardware_goals']
                world.command_bounds = holding,self.robot.synchronized_command(goal,mapping)
                reason = None
                for q in path:
                    check()
                    for sample in self.recovery_envelope(start,q,holding):
                        reason = world.reason(sample)
                        if reason:
                            break
                    command = self.robot.synchronized_command(q,mapping)
                    if not reason:
                        reason = world.reason(command,command=True)
                    if reason:
                        break
                if not reason:
                    return path,world,mapping
                first_refusal = first_refusal or reason
                # A farther rotation on this side contains the same failed edge.
                rejected.add(side)
                if len(rejected)==2:
                    break
        raise ValueError('no checked overview change: '+str(first_refusal or 'cube remains occluded'))

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
                released = (any(m['torque_enabled'] for m in state.get('motor_health',{}).values()) and
                            (self.qualification_file.parent/'released-placement.pkl').exists())
                add('RETREAT_CLEARANCE' if released else 'STARTUP_PREPARATION','GEOMETRY' if released else 'STARTUP',
                    'после release фактическая поза не проходит рабочий запас; требуется проверенный выход пустого захвата' if released else
                    'штатная стартовая поза требует prepare_ready до применения рабочих запасов 5°/8 мм; это не отказ калибровки суставов',
                    'путь из свежих энкодеров сохраняет все ограничения и завершается с рабочими зазорами; STOP снимается только вручную')
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

    def recovery_world(self, native_scene, start):
        import copy
        world = self.make_world(native_scene)
        clearances = self.robot.backend.model.pair_clearances(start)
        initial = {name:gap for name,gap in clearances.items() if gap<self.robot.backend.model.gap}
        if any(gap<=0. for gap in initial.values()):
            raise ValueError('recovery starts in a native zero-clearance contact')
        backend = copy.copy(self.robot.backend)
        backend.model = NativeRecoveryMargins(backend.model,initial)
        world.native_checks = ExtractionCollisionWorld(backend,world.ground_scene)
        world.attachment = copy.copy(self.attachment)
        frames = self.robot.backend.model.frames(start)
        box = native_scene.box
        primary = box.primary
        physical = {}
        for lower,upper,label in primary.solid_parts():
            lo,hi = np.asarray(lower).copy(),np.asarray(upper).copy()
            if label=='box floor':
                hi[2] = primary.floor_interval_m[1]
                physical[label] = lo,hi
            if label.startswith('box ') and label.endswith('wall'):
                lo[2] = primary.floor_interval_m[1]
                hi[2] = primary.obstacle_top_m
                if label=='box left wall': hi[0] = primary.inner_min_xy[0]
                if label=='box right wall': lo[0] = primary.inner_max_xy[0]
                if label=='box front wall': hi[1] = primary.inner_min_xy[1]
                if label=='box rear wall': lo[1] = primary.inner_max_xy[1]
                physical[label] = lo,hi
        from so101_pick_place.experiments.box_extraction.mesh_collision import mesh_box_overlap
        link_walls = {}
        for name,parts in world.collision_meshes.items():
            if name in frames:
                points = self.robot.backend._transform_points(frames[name],self.robot.backend.meshes[name])
                if box.collision(points):
                    vertices = np.concatenate([p.triangles.reshape(-1,3) for p in parts])
                    local = self.robot.backend._transform_points(frames[name],vertices)@box.inverse[:3,:3].T+box.inverse[:3,3]
                    for lower,upper,label in box.solid_parts():
                        if mesh_box_overlap(parts,frames[name],box.base_from_box,lower,upper):
                            if label not in physical:
                                raise ValueError(name+': recovery cannot relax '+label)
                            lo0,hi0 = physical[label]
                            gap = float(np.max(np.maximum(lo0-local.max(0),local.min(0)-hi0)))
                            if gap<=0.:
                                raise ValueError(name+': physical wall intersection '+label)
                            link_walls[name,label] = gap
        world.native_checks = RecoveryCollisionWorld(backend,world.ground_scene,link_walls,physical)
        world.recovery_factory = lambda scene:RecoveryCollisionWorld(backend,scene,link_walls,physical)
        walls = {}
        for name,parent,lo,hi,vertices in world.attachment.parts:
            points = self.robot.backend._transform_points(frames[parent],vertices)
            local = points@box.inverse[:3,:3].T+box.inverse[:3,3]
            for lower,upper,label in box.solid_parts():
                if np.all(local.max(0)>=lower) and np.all(local.min(0)<=upper):
                    if label not in physical:
                        raise ValueError(name+': recovery cannot relax '+label)
                    lo0,hi0 = physical[label]
                    gap = float(np.max(np.maximum(lo0-local.max(0),local.min(0)-hi0)))
                    if gap<=0.:
                        raise ValueError(name+': physical wall envelope intersects '+label)
                    walls[name,label] = {'lower':lo0,'upper':hi0,'minimum':gap}
        world.attachment.recovery_walls = walls
        world.recovery_native_minimums = initial
        world.recovery_walls = walls
        world.recovery_link_walls = link_walls
        world.physical_walls = physical
        # Native BVH coordinates are float32 even though FK frames are double.
        # Bound two homogeneous dot products and their subtraction (gamma_n),
        # rather than comparing their results with an exact equality.
        scale = max(float(np.max(np.abs(frame[:3,:3])@np.abs(self.robot.backend.meshes[name]).T+
                                  np.abs(frame[:3,3,None])))
                    for name,frame in frames.items() if name in self.robot.backend.meshes)
        operations = 2*(2*next(iter(frames.values())).shape[0]-1)+1
        unit = np.finfo(np.float32).eps/2
        world.recovery_numeric_resolution = float(operations*unit/(1-operations*unit)*scale)
        world.empty_recovery = True
        return world

    def recovery_gaps(self, world, joints):
        frames = self.robot.backend.model.frames(joints)
        native = self.robot.backend.model.pair_clearances(joints)
        result = {'native: '+name:native[name] for name in world.recovery_native_minimums}
        inverse = world.scene.box.inverse
        for (name,label),minimum in world.recovery_link_walls.items():
            vertices = np.concatenate([p.triangles.reshape(-1,3) for p in world.collision_meshes[name]])
            points = self.robot.backend._transform_points(frames[name],vertices)@inverse[:3,:3].T+inverse[:3,3]
            lo,hi = world.physical_walls[label]
            result[name+': '+label] = float(np.max(np.maximum(lo-points.max(0),points.min(0)-hi)))
        for (name,label),bound in world.recovery_walls.items():
            part = next(p for p in world.attachment.parts if p[0]==name)
            points = self.robot.backend._transform_points(frames[part[1]],part[4])@inverse[:3,:3].T+inverse[:3,3]
            result[name+': '+label] = float(np.max(np.maximum(bound['lower']-points.max(0),points.min(0)-bound['upper'])))
        return result

    def install_recovery_bounds(self, world, lower):
        comparison = {key:max(0.,value-world.recovery_numeric_resolution) for key,value in lower.items()}
        for key in world.recovery_native_minimums:
            world.recovery_native_minimums[key] = comparison['native: '+key]
        for key in world.recovery_link_walls:
            world.recovery_link_walls[key] = comparison[key[0]+': '+key[1]]
        for key,bound in world.recovery_walls.items():
            bound['minimum'] = comparison[key[0]+': '+key[1]]
        world.native_checks.backend.model.minimums = world.recovery_native_minimums
        world.cache.clear()

    def recovery_envelope(self, start, joints, holding_goals, settled=False):
        import itertools,inspect,re
        native = inspect.getsource(ExistingArmDriver._move)
        arrival = float(re.search(r'follow\(poses,.*?within=([\d.]+)',native,re.S).group(1))
        command = self.robot.quantized_command(self.robot.synchronized_command(joints,self.recovery_mapping))
        steps = self.robot.encoder_steps()
        names = list(start)
        limits = []
        for name in names:
            delta = joints[name]-start[name]
            lag = arrival if settled else self.planner.tracking_error_deg
            behind = float(np.clip(joints[name]-np.sign(delta)*lag,min(start[name],joints[name]),max(start[name],joints[name])))
            reaction = start[name]+command[name]-holding_goals[name]
            limits.append((min(joints[name],behind,reaction)-steps[name]/2,
                           max(joints[name],behind,reaction)+steps[name]/2))
        for values in itertools.product(*limits):
            yield dict(zip(names,values))

    def validate_recovery_envelope(self, world, start, points, holding, policy, planning_cancel):
        baseline = self.recovery_gaps(world,start)
        minima = {key:float('inf') for key in baseline}
        previous = None
        for index,point in enumerate(points):
            bounds = {key:float('inf') for key in baseline}
            for sample in self.recovery_envelope(start,point,holding):
                planning_cancel()
                refusal = world.reason(sample,**policy)
                if refusal:
                    self.recovery_report.update(rejected_waypoint=index,rejected_nominal_pose=point,
                        rejected_response_pose=sample,rejected_response_gaps_m=self.recovery_gaps(world,sample))
                    raise ValueError('uncertainty: '+refusal)
                gaps = self.recovery_gaps(world,sample)
                bounds = {key:min(bounds[key],gaps[key]) for key in bounds}
            capped = {key:min(value,self.robot.backend.model.gap if key.startswith('native: ') else world.scene.box.clearance_m) for key,value in bounds.items()}
            if previous is not None and any(capped[key]<previous[key]-world.recovery_numeric_resolution for key in capped):
                raise ValueError('predicted deficient gap decreases')
            minima = {key:min(minima[key],bounds[key]) for key in bounds}
            previous = capped
        return minima

    def recovery_candidate_is_valid(self, world, start, joints, holding, policy):
        # Reject the established failure cheaply before native mesh search:
        # a stopped compensated target may not cross the fixed finger plane.
        placed = policy['placed_cube']
        for sample in self.recovery_envelope(start,joints,holding):
            for q in (sample,):
                if self.planner._fixed_jaw_clearance(q,placed)<0.:
                    frame = self.robot.backend.model.frames(q)['gripper_link']
                    points = self.robot.backend._transform_points(frame,self.robot.backend.meshes['gripper_link'])
                    if np.all(points.max(0)>=placed.vertices_m.min(0)) and np.all(points.min(0)<=placed.vertices_m.max(0)):
                        return False
        return world.reason(joints,**policy) is None and world.reason(
            self.robot.synchronized_command(joints,self.recovery_mapping),**policy) is None

    def plan_invalid_start(self, native_scene, state, placed, planning_cancel):
        """Calculate a finite native recovery; no actuation and no STOP reset."""
        import inspect,re
        start = dict(state['positions'])
        if not self.robot.audit_state(state)['hard_valid']:
            raise ValueError('RECOVERY_FROM_INVALID_START: hard native constraints fail')
        holding = {name:motor['goal'] for name,motor in state['motor_health'].items()}
        self.recovery_mapping = self.robot.synchronize_reference(state,self.planner.command_sag)
        world = self.recovery_world(native_scene,start)
        target = next(key for key,cube in native_scene.cubes.items() if np.linalg.norm(cube.center_m-placed.center_m)<np.finfo(float).eps)
        policy = {'target':target,'jaw_contact':True,'placed_cube':placed}
        normal = self.make_world(native_scene)
        baseline = self.recovery_gaps(world,start)
        # Every path starts with this same native gravity-compensated packet.
        # Reject it once before searching endpoints; no later waypoint can fix
        # a new contact in the initial response envelope.
        self.recovery_report = {'status':'CHECKING','phase':'RECOVERY_FROM_INVALID_START',
            'start':start,'goal':None,'initial_gaps_m':baseline,
            'initial_gaps_with_execution_error_m':None,
            'holding_hardware_goals':holding,
            'first_hardware_command':self.recovery_mapping['first_command'],
            'reference_synchronization':self.recovery_mapping,
            'actuator_commands_sent':False,'stop_reset':False}
        # A new STOP must also be safe at the very first recovery point.
        # freeze() sends the encoder snapshot as raw goals without compensation.
        # Replay the already measured post-STOP hold deflection as an uncertainty
        # scenario; it is not a new calibration or an assumed exact prediction.
        stop_goals = self.robot.quantized_command(start)
        settling = {name:start[name]-holding[name] for name in start}
        stop_pose = {name:stop_goals[name]+settling[name] for name in start}
        stop_gaps = self.recovery_gaps(world,stop_pose)
        self.recovery_report.update(native_stop_raw_goals=stop_goals,
            measured_hold_deflection=settling,stop_settling_scenario=stop_pose,stop_settling_gaps_m=stop_gaps)
        contact = next((key for key,value in stop_gaps.items() if value<=0.),None)
        supported = self.recovery_mapping.get('supported_stop_packet_checked') is True
        if supported:
            self.recovery_report.update(stop_policy='last_checked_hardware_goals',
                default_freeze_rejected=contact is not None,
                supported_stop_goals=self.recovery_mapping['supported_stop_goals'],
                stop_response_model='trajectory aborted; last checked targets retained; settling enclosed by native tracking and command-response envelope',
                physical_supported_stop_tested=False)
        if contact and not supported:
            self.recovery_report.update(status='BLOCKED',code='STOP_SETTLING_CLEARANCE',pair=contact,
                reason='native STOP rewrites goals to encoders; observed hold deflection reaches '+contact)
            raise ValueError('RECOVERY_FROM_INVALID_START: '+self.recovery_report['reason'])
        lower = dict(baseline)
        initial_states = list(self.recovery_envelope(start,start,holding))
        for q in initial_states:
            values = self.recovery_gaps(world,q)
            lower = {key:min(lower[key],values[key]) for key in lower}
        if any(value<=0. for value in lower.values()):
            raise ValueError('initial execution uncertainty reaches physical contact: '+str(lower))
        # Keep the physical bounds; account for native float32 arithmetic only.
        self.install_recovery_bounds(world,lower)
        self.recovery_report['initial_gaps_with_execution_error_m'] = lower
        self.recovery_report['floating_point_resolution_m'] = world.recovery_numeric_resolution
        for sample in initial_states:
            planning_cancel()
            refusal = world.reason(sample,**policy)
            if refusal:
                self.recovery_report.update(status='BLOCKED',pair=refusal,
                    reason='initial native command response: '+refusal,
                    rejected_response_pose=sample,
                    initial_fixed_jaw_gap_m=self.planner._fixed_jaw_clearance(start,placed),
                    rejected_fixed_jaw_gap_m=self.planner._fixed_jaw_clearance(sample,placed))
                raise ValueError('RECOVERY_FROM_INVALID_START: initial native command response: '+refusal)
        source = inspect.getsource(ExistingArmDriver._move)
        arrival = float(re.search(r'follow\(poses,.*?within=([\d.]+)',source,re.S).group(1))
        previous_cancel = self.planner.check_cancel
        previous_check = self.check_cancel
        previous_command = self.planner._command_pose
        self.planner.check_cancel = planning_cancel
        self.check_cancel = planning_cancel
        failures = []
        selected = None
        try:
            for label,goal,indices,limits in self.recovery_goals(start,world,normal,policy):
                if selected is None:
                    self.report('checking','recovery '+label)
                    try:
                        from so101_pick_place.experiments.box_extraction.search import rrt_connect
                        names = tuple(start)
                        def expand(values):
                            q = dict(start)
                            q.update({names[index]:float(value) for index,value in zip(indices,values)})
                            return q
                        route = rrt_connect([start[names[i]] for i in indices],
                            [goal[names[i]] for i in indices],limits,
                            lambda values:self.recovery_candidate_is_valid(world,start,expand(values),holding,policy),
                            check_cancel=planning_cancel)
                        points = tuple(expand(values) for values in route.points)
                        minima = self.validate_recovery_envelope(world,start,points,holding,policy,planning_cancel)
                        for sample in self.recovery_envelope(start,points[-1],holding,settled=True):
                            refusal = normal.reason(sample,**policy)
                            if refusal:
                                raise ValueError('ordinary endpoint envelope: '+refusal)
                        for q in points:
                            reference = self.robot.control_reference(q,self.recovery_mapping)
                            command = self.robot.synchronized_command(q,self.recovery_mapping)
                            if self.robot.quantized_command(reference)==self.robot.quantized_command(command):
                                raise ValueError('normal packet is not distinguishable from an uncompensated stop')
                            refusal = world.reason(command,**policy)
                            if refusal:
                                raise ValueError('native compensated command: '+refusal)
                        selected = points,minima
                        self.recovery_report.update(route_family=label,rrt_iterations=route.trace['iterations'])
                    except ValueError as exc:
                        failures.append(label+': '+str(exc))
                        self.recovery_report['candidate_failures'] = list(failures)
                        self.report('rejected',failures[-1])
                if selected is not None:
                    break
            if selected is None:
                raise ValueError('; '.join(failures) or 'no goal outside the measured box is reachable')
            points,minima = selected
            self.planner._command_pose = lambda q:self.robot.synchronized_command(q,self.recovery_mapping)
            previous = ExtractionPlan(native_scene.revision,target,(Phase('retract',points,placed_cube=placed),),
                                      np.empty((0,3)),placed.center_m,())
            self.report('checking','recovery Park from the calculated retreat endpoint')
            park = self.plan_park(points[-1],previous,normal,recovery=True)
            park_minima = self.validate_recovery_envelope(world,start,park.joints,holding,policy,planning_cancel)
            minima = {key:min(minima[key],park_minima[key]) for key in minima}
        except ValueError as exc:
            self.recovery_report.update(status='BLOCKED',reason=str(exc))
            raise ValueError('RECOVERY_FROM_INVALID_START: '+str(exc)) from exc
        finally:
            self.planner.check_cancel = previous_cancel
            self.check_cancel = previous_check
            self.planner._command_pose = previous_command
        phase = Phase('RECOVERY_FROM_INVALID_START',points,jaw_contact=True,placed_cube=placed)
        report = self.recovery_report|{'goal':points[-1],'minimum_gaps_with_execution_error_m':minima,
            'joint_path':points,'park_path':park.joints,'park_goal':park.joints[-1],
            'ordinary_native_gap_m':self.robot.backend.model.gap,
            'ordinary_wall_gap_m':native_scene.box.clearance_m,'encoder_steps':self.robot.encoder_steps(),
            'tracking_error_deg':self.planner.tracking_error_deg,'native_arrival_tolerance_deg':arrival,
            'status':'READY_FOR_ATTENDED_CONFIRMATION','cable':'operator-supervised; no independent swept cable model',
            'actuator_commands_sent':False,'stop_reset':False,'endpoint_requires_fresh_encoder_confirmation':True}
        self.recovery_report = report
        return phase,world,normal,report

    def recovery_goals(self, start, world, normal, policy):
        """Derive lift goals from FK; use the existing native joint-space search."""
        model = self.robot.backend.model
        names = tuple(start)
        steps = self.robot.encoder_steps()
        rim = float((world.scene.box.base_from_box@np.r_[0.,0.,world.scene.box.obstacle_top_m,1.])[2])
        def lowest(q):
            frames = model.frames(q)
            heights = [self.robot.backend._transform_points(frames[link],self.robot.backend.meshes[link])[:,2].min()
                       for link in ('gripper_link','moving_jaw_so101_v1_link')]
            heights.extend(self.robot.backend._transform_points(frames[parent],vertices)[:,2].min()
                           for name,parent,lo,hi,vertices in world.attachment.parts)
            return float(min(heights))
        initial_height = lowest(start)
        gradient = {}
        for index in (1,2,3):
            name = names[index]
            gradient[index] = (lowest(dict(start,**{name:start[name]+steps[name]}))-initial_height)/steps[name]
        cube = policy['placed_cube']
        pan = names[0]
        probe = dict(start,**{pan:start[pan]+steps[pan]})
        outward = np.sign(self.planner._fixed_jaw_clearance(probe,cube)-self.planner._fixed_jaw_clearance(start,cube))
        pan_range = (start[pan],model.ranges[0][1]-model.margin) if outward>=0 else (model.ranges[0][0]+model.margin,start[pan])
        for label,axes in (('coordinated shoulder/elbow',(1,2)),('shoulder lift',(1,)),('elbow lift',(2,)),('wrist-assisted lift',(1,2,3))):
            direction = {index:gradient[index] for index in axes}
            peak = max(abs(value) for value in direction.values())
            if peak>np.finfo(float).eps:
                direction = {index:value/peak for index,value in direction.items()}
                spans = [(model.ranges[index][1]-model.margin-start[names[index]])/value if value>0 else
                         (model.ranges[index][0]+model.margin-start[names[index]])/value
                         for index,value in direction.items() if value!=0.]
                limit = min(spans)
                far = dict(start,**{names[index]:start[names[index]]+value*limit for index,value in direction.items()})
                goal = None
                for q in self.robot.native_edge_samples(start,far):
                    self.planner.check_cancel()
                    if lowest(q)>=rim+model.gap and normal.reason(q,**policy) is None:
                        reserve = self.planner.tracking_error_deg+max(steps[names[index]] for index in axes)
                        candidate = dict(q,**{names[index]:q[names[index]]+value*reserve for index,value in direction.items()})
                        if normal.reason(candidate,**policy) is None:
                            goal = candidate
                        break
                if goal is not None:
                    indices = (0,*axes)
                    bounds = [pan_range]+[(min(start[names[index]],goal[names[index]]),max(start[names[index]],goal[names[index]])) for index in axes]
                    if 3 in axes:
                        bounds = [pan_range]+[(model.ranges[index][0]+model.margin,model.ranges[index][1]-model.margin) for index in axes]
                    if all(lo<hi for lo,hi in bounds):
                        yield label,goal,indices,np.asarray(bounds)

    def state_is_valid(self, q, phase, plan, world):
        return world.reason(q,target=plan.target_id,jaw_contact=phase.jaw_contact,
            payload_local=plan.payload_local if phase.carrying else None,support_contact=phase.support_contact,
            placed_cube=phase.placed_cube)

    def edge_is_valid(self, start, goal, phase, plan, world):
        if self.planner.command_sag is None:
            raise ValueError('native gravity-compensation setting has not been read')
        for q in self.robot.native_edge_samples(start,goal):
            self.check_cancel()
            command = self.robot.synchronized_command(q,self.recovery_mapping) if phase.name.startswith('RECOVERY_') else self.planner._command_pose(q)
            for kind,pose in (('pose',q),('native command',command)):
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
                if not phase.name.startswith('RECOVERY_'):
                    for q in phase.joints:
                        self.check_cancel()
                        command = self.robot.quantized_command(self.planner._command_pose(q))
                        frozen = self.robot.quantized_command(q)
                        # The factory STOP removes the position bias supporting
                        # gravity. Check that transition before committing to
                        # a grasp/release, rather than discovering it in a box.
                        settled = {j:frozen[j]+q[j]-command[j] for j in q}
                        reason = self.state_is_valid(settled,phase,plan,world)
                        intended = reason in {'gripper_link: cube '+plan.target_id,'moving_jaw_so101_v1_link: cube '+plan.target_id}
                        if intended and phase.name=='grasp' and phase.jaw_contact:
                            reason = world.grasp_contact_reason(settled,plan.target_id)
                        if reason:
                            raise ValueError(phase.name+': native STOP settling: '+reason)
                if phase.name in ('retract','safe_park','RECOVERY_PARK'):
                    for lag in self.planner._lagging_poses(phase.joints):
                        self.check_cancel()
                        for pose in (lag,self.planner._command_pose(lag)):
                            reason = self.state_is_valid(pose,phase,plan,world)
                            if reason:
                                raise ValueError(phase.name+': tracking envelope: '+reason)
            finally:
                world.release_origin = None
        return True

    def preflight_cycle(self, plan, world, require_pick=True):
        """A release is admitted only with an empty-tool exit and safe wait path."""
        names = [phase.name for phase in plan.phases]
        required = (('approach','enter','grasp') if require_pick else ())+('extract_vertical','transfer','lower_to_support','release','retract')
        missing = [name for name in required if name not in names]
        if missing or any(names.index(a)>=names.index(b) for a,b in zip(required,required[1:]) if a in names and b in names):
            raise ValueError('incomplete Pick & Place cycle: '+(', '.join(missing) or 'invalid phase order'))
        retreat = plan.phases[names.index('retract')]
        if retreat.carrying or retreat.placed_cube is None:
            raise ValueError('retreat must use empty jaws and the stationary released cube')
        self.check_path(plan,world)
        park = self.plan_park(retreat.joints[-1],plan,world)
        return park

    def plan_direct_transfer(self, start, goal, world, **policy):
        return self.planner.transport(start,goal,world,**policy)

    def plan_park(self, start, previous, world, recovery=False):
        """Select a low-load recorded working pose; verify its current transition."""
        placed = next((p.placed_cube for p in reversed(previous.phases) if p.placed_cube is not None),None)
        policy = {'target':previous.target_id,'placed_cube':placed}
        reason = world.reason(start,**policy)
        if reason:
            raise ValueError('Park needs completed retreat and valid start: '+reason)
        candidates = []
        root = self.qualification_file.parent
        for name,key in (('contact-free-trial.json','start'),('startup-trial.json','actual')):
            path = root/name
            if path.exists():
                record = json.loads(path.read_text())
                if record.get('passed') or record.get('physical_completed'):
                    candidates.append(dict(record[key],gripper=start['gripper']))
        def cost(q):
            torque = np.asarray(self.robot.backend.model.pose(q).torque)[1:3]
            return float(torque@torque),float(np.linalg.norm(self.robot.from_native(q)-self.robot.from_native(start)))
        if not candidates:
            raise ValueError('Park: no physically checked waiting configuration is recorded')
        if cost(start)[0]<=min(cost(q)[0] for q in candidates):
            candidates.append(start)
        failures = []
        for goal in sorted(candidates,key=cost):
            reason = world.reason(goal,**policy)
            if reason:
                failures.append(reason)
            else:
                try:
                    path,_ = self.planner.transport(start,goal,world,**policy)
                    phase = Phase('RECOVERY_PARK' if recovery else 'safe_park',path,placed_cube=placed)
                    self.check_path(replace(previous,phases=(phase,)),world)
                    return phase
                except ValueError as exc:
                    failures.append(str(exc))
        raise ValueError('no safe low-load Park transition: '+'; '.join(failures[-3:]))

    def park_arm(self, previous, world, phase=None):
        current = self.robot.read_state()['positions']
        if phase is None:
            phase = self.plan_park(current,previous,world)
        elif any(abs(current[j]-phase.joints[0][j])>self.robot.encoder_steps()[j] for j in current):
            raise ValueError('Park start changed since calculation; recalculate from encoders')
        else:
            phase = replace(phase,joints=(current,)+tuple(phase.joints[1:]))
        target = phase.joints[-1]
        self.check_path(replace(previous,phases=(phase,)),world)
        if any(abs(target[j]-current[j])>self.robot.encoder_steps()[j] for j in current):
            self.execute_trajectory(phase,previous,world)
        actual = self.robot.read_state()['positions']
        reason = world.reason(actual,target=previous.target_id,placed_cube=phase.placed_cube)
        error = max(abs(actual[j]-target[j]) for j in actual if j!='gripper')
        if reason or error>self.planner.tracking_error_deg:
            raise RuntimeError('Park endpoint: '+str(reason or f'following error {error:.3f} deg'))
        torque = np.asarray(self.robot.backend.model.pose(actual).torque)
        return {'status':'PARKED','actual':actual,'target':target,'following_error_deg':error,
                'shoulder_elbow_gravity_torque_nm':torque[1:3].tolist(),'torque_disabled':False}

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
            if phase.name.startswith('RECOVERY_'):
                if self.recovery_mapping.get('supported_stop_packet_checked') is not True:
                    raise RuntimeError('recovery requires a checked load-supported STOP packet')
                import copy
                original = self.driver.backend
                backend = copy.copy(original)
                backend.driver_command_goal = lambda q,sag:self.robot.synchronized_command(q,self.recovery_mapping)
                self.driver.backend = backend
                self.executing_recovery = True
                self.recovery_feedback_at = 0.
                self.recovery_observed_lower = None
                try:
                    with self.robot.synchronized_follow(self.recovery_mapping):
                        self.driver.move_phase(phase,plan,world,self.check_cancel)
                finally:
                    self.executing_recovery = False
                    self.driver.backend = original
            else:
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
        if self.executing_recovery:
            record = dict(record,goal=self.robot.control_reference(record['goal'],self.recovery_mapping,inverse=True))
        if self.startup_world is not None:
            command = (self.robot.synchronized_command(record['goal'],self.recovery_mapping) if self.executing_recovery else
                       self.robot.backend.driver_command_goal(record['goal'],self.robot.arm.raw.sag))
            for kind,q in (('goal',record['goal']),('pos',record['pos']),
                           ('command',command)):
                reason = self.startup_world.reason(q,command=kind=='command')
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
        if self.executing_recovery and hasattr(self.driver.world,'recovery_native_minimums'):
            self.check_recovery_feedback(record['pos'],self.driver.world)

    def check_recovery_feedback(self, joints, world):
        """Stop if measured, encoder-bounded deficient clearances worsen."""
        import itertools
        if time.monotonic()-self.recovery_feedback_at<self.robot.health_interval_s:
            return
        steps = self.robot.encoder_steps()
        names = tuple(joints)
        limits = [(joints[j]-steps[j]/2,joints[j]+steps[j]/2) for j in names]
        samples = [self.recovery_gaps(world,dict(zip(names,values))) for values in itertools.product(*limits)]
        low = {key:min(s[key] for s in samples) for key in samples[0]}
        high = {key:max(s[key] for s in samples) for key in samples[0]}
        previous = self.recovery_observed_lower or self.recovery_report['initial_gaps_with_execution_error_m']
        for key,value in previous.items():
            if high[key]<value-world.recovery_numeric_resolution:
                raise RuntimeError('recovery measured clearance decreases: '+key)
        self.recovery_observed_lower = {key:max(previous[key],min(value,
            self.robot.backend.model.gap if key.startswith('native: ') else world.scene.box.clearance_m)) for key,value in low.items()}
        self.recovery_feedback_at = time.monotonic()

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
    def __init__(self,robot,scene,geometry,start,goal,profile_path,strict=False):
        from scipy.spatial import ConvexHull
        from .diagnostics import minimum_clearance
        from so101_pick_place.experiments.box_extraction.mesh_collision import load_collision_meshes
        self.robot,self.geometry = robot,geometry
        self.strict = strict
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
        self.partial_cubes = self.bound_partial_cubes(scene,geometry)
        self.partial_clearance = {}
        self.command_bounds = None
        self.cache = {}

    @staticmethod
    def bound_partial_cubes(scene,geometry):
        """Bound an occluded known-size tabletop cube without guessing its centre."""
        from .vision import color_masks
        from box_extract.scene_builder import SceneParameters
        frame = scene.frames['overview']
        masks = color_masks(frame.image)
        bounds = []
        side = float(geometry.known_cube['cube_size_m'])
        table = geometry.table_z
        limits = SceneParameters()
        for color in sorted({o.get('color','unknown') for o in scene.state.obstacles}):
            if color not in masks:
                raise ValueError('unclassified obstacle in the observed workspace')
            contours,_ = cv2.findContours(masks[color],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            def in_workspace(contour):
                center = np.mean(contour.reshape(-1,2),axis=0)
                point = geometry.intersect_ray_with_plane(geometry.pixel_ray_in_base(center,frame),([0,0,1],-table))
                return point[0]>=limits.workspace_min_x_m and np.linalg.norm(point[:2])<=limits.workspace_max_radius_m
            contours = [c for c in contours if cv2.contourArea(c)>450. and in_workspace(c)]
            if len(contours)!=1 or any(o.color==color for o in scene.state.objects.values()):
                raise ValueError(color+': no unique overview fragment to bound the occluded cube')
            pixels = cv2.convexHull(contours[0]).reshape(-1,2)
            # Box-contained fragments cannot use the tabletop height. Boxes
            # are already opaque keepouts, but do not silently infer a cube pose.
            if any(cv2.pointPolygonTest(np.float32(b.evidence['polygon']),tuple(map(float,p)),False)>=0
                   for b in scene.state.boxes.values() for p in pixels):
                raise ValueError(color+': fragment may be supported by a box')
            points = np.array([[geometry.intersect_ray_with_plane(geometry.pixel_ray_in_base(p,frame),([0,0,1],-z))
                               for z in (table,table+side)] for p in pixels])
            diameter = np.sqrt(2.)*side
            lo = np.r_[points.min(1).max(0)[:2]-diameter,table]
            hi = np.r_[points.max(1).min(0)[:2]+diameter,table+side]
            x,y,w,h = cv2.boundingRect(contours[0])
            bounds.append((color,lo,hi,np.int32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])))
        return bounds

    def reason(self,q,command=False,**policy):
        key = (command,)+tuple(q[j] for j in self.start)
        if key not in self.cache:
            if len(self.cache)>8192:
                self.cache.clear()
            self.cache[key] = self._reason(q,command)
        return self.cache[key]

    def _reason(self,q,command):
        from so101_pick_place.experiments.box_extraction.mesh_collision import mesh_box_overlap
        robot,model = self.robot,self.robot.backend.model
        start,goal = self.command_bounds if command and self.command_bounds is not None else (self.start,self.goal)
        for name,value in q.items():
            low,high = sorted((start[name],goal[name]))
            # Native sag can move the elbow inwards; no outward range override.
            allowance = 2. if name=='elbow_flex' else .8
            if not low-allowance<=value<=high+.8:
                return name+': outside bounded startup corridor'
        pose = model.pose(q)
        problems = model.problems(pose) if self.strict else model.problems(pose,margin=0.)
        if problems:
            return 'hard constraints: '+str(problems)
        collision = model.collides(pose) if self.strict else model.collides(pose,gap=max(0.,self.initial_gap-.0002))
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
            lo,hi = points.min(0),points.max(0)
            if name not in ('base_link','shoulder_link') and lo[2]<model.table_z+(model.gap if self.strict else .004):
                return name+': table clearance'
            for color,lower,upper,_ in self.partial_cubes:
                gap = float(np.maximum(lower-hi,lo-upper).max())
                self.partial_clearance[color] = min(self.partial_clearance.get(color,float('inf')),gap)
                if gap<model.gap:
                    return name+': partially visible '+color+' cube clearance'
            hull = cv2.convexHull(np.float32(self.geometry.project(points))).astype(np.int32)
            x,y,w,h = cv2.boundingRect(hull)
            x0,y0 = max(0,x),max(0,y)
            x1,y1 = min(self.mask.shape[1],x+w),min(self.mask.shape[0],y+h)
            region = self.mask[y0:y1,x0:x1] if x1>x0 and y1>y0 else None
            if region is not None and np.any(region):
                mask = np.zeros_like(region)
                cv2.fillConvexPoly(mask,hull-np.int32([x0,y0]),255)
                if np.any(cv2.bitwise_and(mask,region)):
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
            factory = getattr(self,'recovery_factory',None)
            self.grasp_worlds[target] = factory(other) if factory is not None else ExtractionCollisionWorld(self.native_checks.backend,other)
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
        if native is None and getattr(self,'empty_recovery',False) and placed is not None:
            from so101_pick_place.experiments.box_extraction.mesh_collision import mesh_cube_overlap
            frames = self.robot.backend.model.frames(joints)
            for link in ('gripper_link','moving_jaw_so101_v1_link'):
                if mesh_cube_overlap(self.collision_meshes[link],frames[link],placed,0.):
                    return link+': new contact with released cube '+str(target)
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
