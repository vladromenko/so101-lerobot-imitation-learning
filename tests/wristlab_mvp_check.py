"""Control-contract checks on recorded feedback; no bus, cameras or robot simulation.

Run with the project's hardware Python. Actuator calls are recorded, never sent.
This verifies software failure handling, not physical manipulation.
"""
import json
import hashlib
from pathlib import Path
import sys
import time
import threading
import tempfile
from collections import deque
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wristlab.actions import Actions
from wristlab.motion import Motion,AccessoryWorld
from wristlab.robot import Robot
from wristlab.geometry import Geometry
from wristlab.scene import Scene
from wristlab.recovery import Recovery
from wristlab.runtime import Runtime
from wristlab import tasks,experiments
from wristlab.types import ObjectState,Placement,SceneState,ActionResult,Frame
from so101_pick_place.geometry.cuboid_vision import cube_vertices
from so101_pick_place.geometry.spatial_cube import SpatialCube
from so101_pick_place.experiments.box_extraction.planner import Phase,ExtractionPlan
from so101_pick_place.experiments.box_extraction.scene import SupportPatch,ExtractionScene


Q = json.loads((ROOT/'runtime/robot-audit.json').read_text())['joints_native']
SIDE = json.loads(Path('/home/jetsonnx/Projects/so101_autonomous_pick_place/config/grasp_tcp_calibration.json').read_text())['cube_size_m']


def object_at(key,color,x,y,z=None):
    pose = np.eye(4);pose[:3,3] = [x,y,SIDE/2 if z is None else z]
    return ObjectState(key,color,None,pose,np.full(3,SIDE),.98,time.time(),
                       evidence={'metric_qualified':True,'position_sigma_m':[.001]*3})


def action_contract():
    obj = object_at('recorded','blue',.26,-.077)
    vertices = cube_vertices(obj.pose[:2,3],SIDE,0.,0.)
    cube = SpatialCube(obj.pose[:3,3],SIDE,np.eye(3),vertices,dict(obj.evidence))
    scene = Scene(NS());scene.state = SceneState({obj.id:obj})
    scene.native_scene = Mock(return_value=ExtractionScene('recorded',time.time(),NS(),{obj.id:cube},NS(),(),True))
    offset = np.array([.001,0.,0.])
    observed_cube = SpatialCube(cube.center_m+offset,SIDE,np.eye(3),vertices+offset,dict(obj.evidence))
    scene.native_object = Mock(return_value=observed_cube)
    state = {'positions':dict(Q),'gripper':Q['gripper'],'timestamp':time.time()}
    robot = NS(read_state=Mock(return_value=state),arm=NS(read=Mock(return_value=(dict(Q),{},{}))),
        backend=NS(model=NS(frames=lambda q:{'gripper_frame_link':np.eye(4)})))
    local = vertices-vertices.mean(0)
    def phase(name):
        return Phase(name,(dict(Q),dict(Q)),name in ('extract_vertical','release'),
                     placed_cube=cube if name=='retract' else None)
    remainder = ('extract_vertical','transfer','lower_to_support','release','retract')
    plan = ExtractionPlan('recorded',obj.id,tuple(phase(s) for s in ('open','approach','enter','grasp')+remainder),
                          local,np.array([.3,0.,SIDE/2]),())
    rest = ExtractionPlan('recorded',obj.id,tuple(phase(s) for s in remainder),
                          local,np.array([.3,0.,SIDE/2]),())
    driver = NS(confirm_grasp=Mock(return_value=local),release_confirmed=Mock(),confirm_retention=Mock(),retention=NS(armed=True))
    motion = NS(qualification_reasons=Mock(return_value=[]),make_world=Mock(return_value=NS()),
        check_path=Mock(),check_entry_transition=Mock(),contact_free_passed=True,driver=driver,planner=NS(),execute_trajectory=Mock(),
        replan_held=Mock(return_value=(rest,NS())),preflight_cycle=Mock(return_value=phase('safe_park')),
        state_is_valid=Mock(return_value=None))
    rt = NS(scene=scene,robot=robot,motion=motion,metrics={},report=Mock(),refresh=Mock(),
        check_cancel=Mock(),wait_pause=Mock(),ensure_driver=Mock(),stop=Mock(),recovery=Recovery(),
        mode='relocate',physical_relocate_count=0,cameras=NS(check=Mock()),stop_event=threading.Event(),
        park_qualified_file=NS(exists=lambda:True),park_arm=Mock())
    action = Actions(rt)
    action.capture_grip_view = Mock(return_value={'frame_id':'anchor'})
    action.verify_pick = Mock(return_value=True)
    observed_pose = obj.pose.copy();observed_pose[:3,3] += offset
    action.verify_placement = Mock(return_value=(True,observed_pose))
    placement = Placement(obj.pose.copy(),'box-1')
    alignment = {'observed_top_m':(obj.pose[:3,3]+[0.,0.,SIDE/2]).tolist()}
    return obj,rt,action,placement,plan,alignment


class RelocateContract(unittest.TestCase):
    def test_active_perception_is_bounded_and_replans_approach(self):
        setup=action_contract();obj,rt,action,placement,plan,alignment=setup
        rt.refine_object=Mock(return_value=obj)
        with patch('wristlab.actions.choose_grasp',return_value=(None,plan,NS())) as choose,patch(
            'wristlab.actions.visual_alignment',side_effect=[ValueError('view occluded'),alignment]):
            result=action.relocate(obj.id,placement)
        self.assertTrue(result.success,result.reason)
        rt.refine_object.assert_called_once()
        self.assertEqual(choose.call_count,3)
        self.assertEqual([c.args[0].name for c in rt.motion.execute_trajectory.call_args_list].count('grasp'),1)

    def test_active_perception_does_not_repeat_same_failed_alignment(self):
        setup=action_contract();obj,rt,action,placement,plan,_=setup
        rt.refine_object=Mock(return_value=obj)
        with patch('wristlab.actions.choose_grasp',return_value=(None,plan,NS())),patch(
            'wristlab.actions.visual_alignment',side_effect=ValueError('target still occluded')):
            result=action.relocate(obj.id,placement)
        self.assertFalse(result.success)
        rt.refine_object.assert_called_once()
        self.assertNotIn('grasp',[c.args[0].name for c in rt.motion.execute_trajectory.call_args_list])
        rt.stop.assert_not_called()
    def invoke(self, setup, **options):
        obj,rt,action,placement,plan,alignment = setup
        with patch('wristlab.actions.choose_grasp',return_value=(None,plan,NS())),patch('wristlab.actions.visual_alignment',return_value=alignment):
            result = action.relocate(obj.id,placement,**options)
        return result

    def test_full_contract_prechecks_exit_and_parks_using_observed_cube(self):
        setup = action_contract();obj,rt,action,*_ = setup
        result = self.invoke(setup)
        self.assertTrue(result.success,result.reason)
        self.assertNotIn(obj.id,rt.motion.replan_held.call_args.args[0].cubes)
        names = [c.args[0].name for c in rt.motion.execute_trajectory.call_args_list]
        self.assertLess(names.index('extract_vertical'),names.index('release'))
        action.verify_pick.assert_called_once()
        action.verify_placement.assert_called_once()
        self.assertGreaterEqual(rt.motion.preflight_cycle.call_count,2)
        rt.park_arm.assert_called_once()
        self.assertIs(rt.motion.make_world.call_args.args[0].cubes[obj.id],rt.scene.native_object.return_value)
        park_plan = rt.park_arm.call_args.args[0]
        self.assertIs(next(p.placed_cube for p in park_plan.phases if p.name=='retract'),rt.scene.native_object.return_value)
        self.assertTrue(result.evidence['retreat_completed'])
        self.assertTrue(result.evidence['parked'])
        self.assertEqual(rt.physical_relocate_count,1)

    def test_unconfirmed_holding_stops_before_transfer_and_release(self):
        setup = action_contract();_,rt,action,*_ = setup
        action.verify_pick.return_value = False
        result = self.invoke(setup)
        self.assertFalse(result.success)
        self.assertTrue(result.evidence['critical'])
        self.assertNotIn('release',[c.args[0].name for c in rt.motion.execute_trajectory.call_args_list])
        rt.stop.assert_called_once()
        self.assertTrue(rt.stop.call_args.kwargs['hardware_fault'])
        self.assertIsNotNone(rt.scene.held)

    def test_failed_placement_observation_does_not_regrasp_retained_coordinates(self):
        setup=action_contract();_,rt,action,*_=setup
        action.verify_placement.return_value=(False,None)
        result=self.invoke(setup)
        self.assertFalse(result.success)
        self.assertTrue(result.evidence['motion_started'])
        self.assertEqual([c.args[0].name for c in rt.motion.execute_trajectory.call_args_list].count('grasp'),1)
        rt.stop.assert_not_called()

    def test_qualification_failure_never_prepares_or_moves_driver(self):
        setup = action_contract();_,rt,*_ = setup
        rt.motion.qualification_reasons.return_value = ['start clearance below 8mm']
        result = self.invoke(setup)
        self.assertFalse(result.success)
        rt.ensure_driver.assert_not_called()
        rt.motion.execute_trajectory.assert_not_called()
        rt.stop.assert_not_called()
        self.assertFalse(rt.scene.reserved)

    def test_dry_run_cannot_write_actuators_or_confirm_grasp(self):
        setup = action_contract();_,rt,*_ = setup
        result = self.invoke(setup,dry=True)
        self.assertTrue(result.success)
        self.assertFalse(result.evidence['physical_verified'])
        self.assertEqual([p['name'] for p in result.evidence['checked_phases']][-2:],['retract','safe_park'])
        self.assertEqual(result.evidence['checked_phases'][-1]['goal_deg'],Q)
        rt.ensure_driver.assert_not_called()
        rt.motion.execute_trajectory.assert_not_called()
        rt.motion.driver.confirm_grasp.assert_not_called()

    def test_powered_goal_transition_rejects_outward_elbow_step_before_motion(self):
        setup = action_contract();_,rt,_,_,plan,_ = setup
        state = rt.robot.read_state()
        state['timestamp'] = time.time()
        actual = state['positions']
        rt.robot.read_motor_health = Mock(return_value={
            name:{'goal':value-(1.32 if name=='elbow_flex' else 0.),'torque_enabled':True,'status_raw':0,'moving':False}
            for name,value in actual.items()})
        rt.robot.quantized_command = dict
        rt.robot.native_edge_samples = Robot.native_edge_samples
        rt.motion.robot = rt.robot
        rt.motion.planner._command_pose = dict
        rt.motion.check_cancel = rt.check_cancel
        rt.motion.state_is_valid.side_effect = lambda q,*_: (
            'elbow_flex outside its range' if q['elbow_flex']>actual['elbow_flex']+.2 else None)
        rt.motion.check_entry_transition.side_effect = lambda *args: Motion.check_entry_transition(rt.motion,*args)
        result = self.invoke(setup)
        self.assertFalse(result.success)
        self.assertIn('native holding transition: elbow_flex outside its range',result.reason)
        rt.motion.preflight_cycle.assert_not_called()
        rt.motion.execute_trajectory.assert_not_called()

    def test_missing_visual_target_after_approach_does_not_latch_hardware_fault(self):
        setup = action_contract();obj,rt,action,placement,plan,_ = setup
        with patch('wristlab.actions.choose_grasp',return_value=(None,plan,NS())),patch('wristlab.actions.visual_alignment',side_effect=ValueError('wrist target is missing')):
            result = action.relocate(obj.id,placement)
        self.assertFalse(result.success)
        self.assertEqual(result.evidence['failure_kind'],'PLANNING_OR_OBSERVATION')
        self.assertFalse(result.evidence['critical'])
        rt.stop.assert_not_called()
        self.assertNotIn('grasp',[call.args[0].name for call in rt.motion.execute_trajectory.call_args_list])

    def test_post_grasp_ik_failure_holds_without_hardware_alarm(self):
        setup = action_contract();_,rt,*_ = setup
        rt.motion.replan_held.side_effect = ValueError('no placement IK')
        result = self.invoke(setup)
        self.assertFalse(result.success)
        self.assertTrue(result.evidence['critical'])
        self.assertFalse(result.evidence['hardware_fault'])
        self.assertIsNotNone(rt.scene.held)
        rt.motion.driver.confirm_retention.assert_called_once()
        rt.stop.assert_not_called()

    def test_operator_stop_is_not_reclassified_as_hardware_fault(self):
        setup = action_contract();_,rt,*_ = setup
        def stop_during_approach(*args):
            rt.stop_event.set()
            raise RuntimeError('STOP')
        rt.motion.execute_trajectory.side_effect = stop_during_approach
        result = self.invoke(setup)
        self.assertEqual(result.evidence['failure_kind'],'OPERATOR_STOP')
        self.assertFalse(result.evidence['hardware_fault'])
        rt.stop.assert_not_called()


class GeometryPolicy(unittest.TestCase):
    def test_dry_preflight_failure_never_moves_to_a_qualified_park(self):
        rt=Runtime.__new__(Runtime)
        rt.stop_event=threading.Event();rt.scene=NS(held=None)
        rt.robot=NS(arm=object(),first_command_at=None);rt.park_qualified_file=NS(exists=lambda:True)
        rt.motion=NS(plan=object(),world=object());rt.park_arm=Mock();rt.persist=Mock()
        rt.report=Mock();rt.check_cancel=Mock();rt.blockers=Mock(return_value=[])
        rt.refresh=Mock(side_effect=ValueError('scene changed before dry preflight'))
        rt.last_plan=None;rt.started=time.monotonic();rt.metrics={}
        rt.execute({'mode':'relocate','dry':True})
        self.assertFalse(rt.results[0]['success'])
        rt.park_arm.assert_not_called()

    def test_overview_measurement_rejects_repeated_frame_during_monitored_hold(self):
        rt = Runtime.__new__(Runtime)
        rt.stop_event = threading.Event()
        rt.check_cancel = Mock()
        rt.monitor_motors = Mock()
        rt.config = {'poll_interval_s':.01}
        state = {'positions':Q,'timestamp':time.time(),'q':np.zeros(5),'gripper':Q['gripper']}
        rt.robot = NS(encoder_steps=lambda:{j:.1 for j in Q},read_state=Mock(return_value=state))
        rt.cameras = NS(max_age=.5,check=Mock(),record_state=Mock(),json=Mock(side_effect=lambda *a,**kw:
            {'frame':{'sequence':1,'captured_at_unix_s':time.time()}}))
        with self.assertRaisesRegex(ValueError,'three fresh distinct frames'):
            rt.measure_overview(Q)
        rt.monitor_motors.assert_called()
        self.assertEqual(rt.cameras.json.call_count,3)

    def test_release_checkpoint_cannot_cross_a_reboot(self):
        import pickle
        with tempfile.TemporaryDirectory() as directory:
            rt=Runtime.__new__(Runtime);rt.output=Path(directory);rt.boot_id='current'
            path=rt.output/'released-placement.pkl'
            path.write_bytes(pickle.dumps({'boot_id':'previous'}))
            with self.assertRaisesRegex(ValueError,'not from this boot'):
                rt.read_release_checkpoint()
            path.write_bytes(pickle.dumps({'boot_id':'current','release_actual':Q}))
            self.assertEqual(rt.read_release_checkpoint()['release_actual'],Q)

    def test_camera_initialization_can_retry_after_native_api_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            rt=Runtime.__new__(Runtime);rt.output=Path(directory);rt.metrics={}
            rt.scene=NS(boxes_semantically_seeded=True);rt.refresh=Mock()
            rt.cameras_initialized=False;rt.camera_retry_at=0.
            rt.cameras=NS(apply_controls=Mock(side_effect=[OSError('connection refused'),{}]),start_capture=Mock())
            rt.initialize_cameras()
            self.assertFalse(rt.cameras_initialized)
            self.assertGreater(rt.camera_retry_at,time.monotonic())
            rt.initialize_cameras()
            self.assertTrue(rt.cameras_initialized)
            rt.cameras.start_capture.assert_called_once()

    def test_overview_view_requires_attended_approval_for_that_proposal(self):
        rt=Runtime.__new__(Runtime);rt.stop_generation=2
        rt.overview_preview={'checked_at':1.,'stop_generation':2}
        rt.refresh=Mock()
        with self.assertRaisesRegex(ValueError,'explicit attended approval'):
            rt.prepare_overview_clearance({'operator_at_stop':True,'proposal_checked_at':3.})
        rt.refresh.assert_not_called()

    def test_encoder_drift_invalidates_an_overview_proposal_before_actuation(self):
        rt=Runtime.__new__(Runtime);rt.stop_generation=2;rt.overview_preview={'checked_at':1.}
        rt.config={'poll_interval_s':1.};rt.report=Mock();rt.refresh=Mock();rt.monitor_motors=Mock()
        rt.scene=NS(state=NS(boxes={'observed':object()}));rt.geometry=NS()
        rt.cameras=NS(check=Mock(),record_state=Mock())
        health={j:{'goal':q,'status_raw':0,'torque_enabled':True} for j,q in Q.items()}
        state={'positions':Q,'motor_health':health,'timestamp':time.time(),'q':np.zeros(5),'gripper':Q['gripper']}
        changed=state|{'positions':Q|{'gripper':Q['gripper']+100./1511.}}
        rt.robot=NS(owns_lease=lambda:True,read_state=Mock(side_effect=[state,changed]),
                    read_motor_health=Mock(return_value=health),arm=NS(follow=Mock()))
        rt.motion=NS(overview_clear_path=Mock(return_value=((Q,Q),NS(),{'hardware_goals':Q})))
        with self.assertRaisesRegex(ValueError,'changed during overview calculation'):
            rt.prepare_overview_clearance({'dry':True})
        self.assertIsNone(rt.overview_preview)
        rt.robot.arm.follow.assert_not_called()

    def test_placement_cannot_be_proved_by_retained_or_old_metric_coordinates(self):
        scene=Scene(NS(wrist=None))
        obj=object_at('target','green',.25,0.)
        placement=Placement(obj.pose.copy(),'table')
        scene.state=SceneState({obj.id:obj},timestamp=20.,evidence={'overview_frame_id':'current'})
        robot=NS(read_state=Mock(return_value={'timestamp':20.,'q':np.zeros(5),'gripper':44.}))
        rt=NS(scene=scene,robot=robot,motion=NS(),refresh=Mock(),check_cancel=Mock(),
              cameras=NS(record_state=Mock(),wait_for_frame=Mock()))
        action=Actions(rt)
        cases=[('occluded',False,21.,'current'),('observed',True,19.,'current'),('observed',True,21.,'previous')]
        with patch('wristlab.actions.time.time',return_value=20.):
            for status,visible,stamp,frame_id in cases:
                obj.status,obj.timestamp=status,stamp
                obj.evidence.update(visible_overview=visible,frame_id=frame_id,spatial_timestamp=stamp)
                self.assertFalse(action.verify_placement(obj,placement)[0])
            obj.status,obj.timestamp='observed',21.
            obj.evidence.update(visible_overview=True,frame_id='current',spatial_timestamp=21.)
            self.assertTrue(action.verify_placement(obj,placement)[0])

    def test_placement_id_reassociation_removes_the_duplicate_obstacle_identity(self):
        scene=Scene(NS(wrist=None))
        target=object_at('target','green',.25,0.)
        seen=object_at('new-id','green',.25,0.);seen.timestamp=21.
        seen.evidence.update(visible_overview=True,frame_id='current',spatial_timestamp=21.)
        scene.state=SceneState({seen.id:seen},boxes={'box':NS(occupied_slots=[seen.id])},
            timestamp=21.,evidence={'overview_frame_id':'current'})
        scene.track_memory[seen.id]=seen;scene.metric_objects[seen.id]=NS()
        robot=NS(read_state=Mock(return_value={'timestamp':20.,'q':np.zeros(5),'gripper':44.}))
        rt=NS(scene=scene,robot=robot,motion=NS(),refresh=Mock(),check_cancel=Mock(),
              cameras=NS(record_state=Mock(),wait_for_frame=Mock()))
        with patch('wristlab.actions.time.time',return_value=20.):
            result=Actions(rt).verify_placement(target,Placement(target.pose.copy(),'table'))
        self.assertTrue(result[0])
        self.assertNotIn('new-id',scene.track_memory)
        self.assertNotIn('new-id',scene.metric_objects)
        self.assertEqual(scene.state.boxes['box'].occupied_slots,['target'])
        self.assertIs(scene.track_memory['target'],scene.state.objects['target'])

    def test_extraction_prioritizes_selected_accessible_cube_and_keeps_top_first(self):
        scene=Scene(NS())
        base=object_at('base','green',.25,0.)
        top=object_at('top','yellow',.25,0.,base.pose[2,3]+SIDE)
        other=object_at('other','blue',.35,0.)
        scene.state.objects={o.id:o for o in (base,top,other)}
        scene.state.boxes={'box':NS(occupied_slots=['top','other','base'])}
        patch_data=SupportPatch(np.array([[.4,-.05],[.5,-.05],[.5,.05],[.4,.05]]),0.,.001,
            {'independent_metric_views':3,'free_surface_verified':True})
        scene.find_free_table_slots=Mock(return_value=[{'id':'table','height_m':0.,'patch':patch_data}])
        self.assertEqual(tasks.next_transfer('extract',scene,'base','box')[0],'top')
        self.assertEqual(tasks.next_transfer('extract',scene,'other','box')[0],'other')

    def test_unknown_color_is_not_silently_assigned_to_the_right_box(self):
        with self.assertRaisesRegex(ValueError,'classified'):
            tasks.box_for_color('unknown',{})

    def test_resumed_native_reference_preserves_all_powered_hardware_goals(self):
        recorded=json.loads((ROOT/'runtime/retreat-park-before-restart.json').read_text())
        robot=Robot('/home/jetsonnx/Projects/so101_autonomous_pick_place',ROOT/'runtime/unit')
        state={'positions':recorded['pos'],'motor_health':recorded['motor_health']}
        synchronization=robot.synchronize_reference(state,3.7)
        expected={n:m['goal'] for n,m in recorded['motor_health'].items()}
        self.assertEqual(synchronization['first_command'],expected)
        native_follow=Mock()
        robot.arm=NS(raw=NS(sag=3.7),follow=native_follow,move_joints=Mock())
        robot.read_motor_health=Mock(return_value=recorded['motor_health'])
        path=[state['positions'],dict(state['positions'],shoulder_pan=state['positions']['shoulder_pan']+.5)]
        with robot.synchronized_follow(synchronization):
            robot.arm.follow(path,duration=1.,lead=False)
        sent=native_follow.call_args.args[0]
        self.assertEqual(robot.quantized_command(robot.backend.driver_command_goal(sent[0],3.7)),expected)
        self.assertAlmostEqual(sent[1]['shoulder_pan']-sent[0]['shoulder_pan'],.5)
        self.assertIs(robot.arm.follow,native_follow)
        self.assertEqual(robot.arm.raw.sag,3.7)
        motion=Motion(robot,lambda:None,lambda *args:None,ROOT/'runtime/qualification.json')
        motion.planner.command_sag=3.7;motion.recovery_mapping=synchronization
        worlds=NS(reason=Mock(return_value=None))
        phase=Phase('RECOVERY_FROM_INVALID_START',(state['positions'],state['positions']))
        plan=NS(target_id='released',payload_local=None)
        motion.edge_is_valid(state['positions'],state['positions'],phase,plan,worlds)
        checked=[call.args[0] for call in worlds.reason.call_args_list]
        self.assertEqual(robot.quantized_command(checked[1]),expected)

    def test_changed_hold_is_rejected_before_the_native_follower(self):
        recorded=json.loads((ROOT/'runtime/retreat-park-before-restart.json').read_text())
        robot=Robot('/home/jetsonnx/Projects/so101_autonomous_pick_place',ROOT/'runtime/unit')
        synchronization=robot.synchronize_reference({'positions':recorded['pos'],'motor_health':recorded['motor_health']},3.7)
        native_follow=Mock();robot.arm=NS(raw=NS(sag=3.7),follow=native_follow,move_joints=Mock())
        recorded['motor_health']['shoulder_lift']['goal']+=2*robot.encoder_steps()['shoulder_lift']
        robot.read_motor_health=Mock(return_value=recorded['motor_health'])
        with robot.synchronized_follow(synchronization):
            with self.assertRaisesRegex(ValueError,'powered hold changed'):
                robot.arm.follow([recorded['pos'],recorded['pos']],lead=False)
        native_follow.assert_not_called()

    def test_supported_stop_scope_is_removed_after_a_native_fault(self):
        recorded=json.loads((ROOT/'runtime/retreat-park-before-restart.json').read_text())
        robot=Robot('/home/jetsonnx/Projects/so101_autonomous_pick_place',ROOT/'runtime/unit')
        state={'positions':recorded['pos'],'motor_health':recorded['motor_health']}
        synchronization=robot.synchronize_reference(state,3.7)
        synchronization['supported_stop_packet_checked']=True
        support=Mock();follow=Mock(side_effect=RuntimeError('native stopped'))
        robot.arm=NS(raw=NS(sag=3.7,supported_stop=support),follow=follow,move_joints=Mock())
        robot.read_motor_health=Mock(return_value=recorded['motor_health'])
        with self.assertRaisesRegex(RuntimeError,'native stopped'):
            with robot.synchronized_follow(synchronization):
                robot.arm.follow([recorded['pos'],recorded['pos']],lead=False)
        self.assertEqual(support.call_args_list[0].args,(synchronization['offset'],))
        self.assertEqual(support.call_args_list[-1].args,(None,))
        self.assertIs(robot.arm.follow,follow)
        self.assertEqual(robot.arm.raw.sag,3.7)

    def test_recovery_execution_cannot_clear_or_bypass_stop(self):
        rt=Runtime.__new__(Runtime);rt.stop_event=threading.Event();rt.stop_event.set()
        rt.cameras=NS(check=Mock());rt.robot=NS(arm=None,enable=Mock())
        with self.assertRaisesRegex(RuntimeError,'STOP'):
            rt.execute_invalid_start({'operator_at_stop':True})
        rt.cameras.check.assert_not_called();rt.robot.enable.assert_not_called()
        self.assertTrue(rt.stop_event.is_set())

    def test_recovery_execution_requires_approval_for_the_checked_proposal(self):
        rt=Runtime.__new__(Runtime);rt.stop_event=threading.Event()
        rt.cameras=NS(check=Mock());rt.robot=NS(arm=None,enable=Mock(),read_state=Mock())
        rt.recovery_proposal={'status':'READY_FOR_ATTENDED_CONFIRMATION','checked_at':1.}
        with self.assertRaisesRegex(ValueError,'explicit attended approval'):
            rt.execute_invalid_start({'operator_at_stop':True,'proposal_checked_at':2.})
        rt.robot.read_state.assert_not_called();rt.robot.enable.assert_not_called()

    def test_measured_clearance_monitor_rejects_a_confirmed_decrease(self):
        motion=Motion.__new__(Motion)
        motion.robot=NS(health_interval_s=0.,encoder_steps=lambda:{'q':.1},backend=NS(model=NS(gap=.008)))
        motion.recovery_feedback_at=0.;motion.recovery_observed_lower=None
        motion.recovery_report={'initial_gaps_with_execution_error_m':{'native: pair':.003}}
        motion.recovery_gaps=lambda world,q:{'native: pair':.004+q['q']*.001}
        world=NS(scene=NS(box=NS(clearance_m=.006)),recovery_numeric_resolution=1e-7)
        motion.check_recovery_feedback({'q':0.},world)
        motion.check_recovery_feedback({'q':0.},world)
        motion.recovery_gaps=lambda world,q:{'native: pair':.003+q['q']*.001}
        with self.assertRaisesRegex(RuntimeError,'measured clearance decreases'):
            motion.check_recovery_feedback({'q':0.},world)

    def test_future_release_is_rejected_if_factory_stop_reaches_a_wall(self):
        recorded=json.loads((ROOT/'runtime/retreat-park-before-restart.json').read_text())
        robot=Robot('/home/jetsonnx/Projects/so101_autonomous_pick_place',ROOT/'runtime/unit')
        motion=Motion(robot,lambda:None,lambda *args:None,ROOT/'runtime/qualification.json')
        motion.planner.command_sag=3.7;q=recorded['pos']
        phase=Phase('release',(q,),jaw_contact=True)
        plan=NS(target_id='released',payload_local=None,phases=(phase,))
        world=NS(reason=lambda pose,**policy:'camera: wall' if pose['shoulder_lift']>q['shoulder_lift']+.5 else None)
        with self.assertRaisesRegex(ValueError,'native STOP settling: camera: wall'):
            motion.check_path(plan,world)
        self.assertIsNone(robot.arm)

    def test_recorded_stop_settling_is_rejected_before_path_search_or_motion(self):
        import pickle
        from wristlab.geometry import AttachmentGeometry
        recorded=json.loads((ROOT/'runtime/retreat-park-before-restart.json').read_text())
        state={'positions':recorded['pos'],'motor_health':recorded['motor_health'],'timestamp':recorded['time']}
        robot=Robot('/home/jetsonnx/Projects/so101_autonomous_pick_place',ROOT/'runtime/unit')
        motion=Motion(robot,lambda:None,lambda *args:None,ROOT/'runtime/qualification.json')
        motion.planner.command_sag=3.7
        motion.attachment=AttachmentGeometry.attended(motion.attachment_path,robot,{'cable_and_hidden_surfaces_clear':True})
        with (ROOT/'runtime/released-placement.pkl').open('rb') as handle:saved=pickle.load(handle)
        placed=next(p.placed_cube for p in reversed(saved['plan'].phases) if p.placed_cube is not None)
        motion.planner.vertical_center=Mock()
        with self.assertRaisesRegex(ValueError,'native STOP rewrites goals'):
            motion.plan_invalid_start(saved['native_scene'],state,placed,lambda:None)
        self.assertEqual(motion.recovery_report['code'],'STOP_SETTLING_CLEARANCE')
        motion.planner.vertical_center.assert_not_called()
        self.assertIsNone(robot.arm)

    def test_incomplete_exit_blocks_before_any_actuator_call(self):
        setup=action_contract();obj,rt,action,placement,plan,_=setup
        rt.motion.preflight_cycle.side_effect=ValueError('retreat: no safe Park route')
        with patch('wristlab.actions.choose_grasp',return_value=(None,plan,NS())):
            result=action.relocate(obj.id,placement)
        self.assertFalse(result.success)
        rt.motion.execute_trajectory.assert_not_called()
        rt.motion.driver.confirm_grasp.assert_not_called()
        rt.park_arm.assert_not_called()

    def test_full_cycle_requires_stationary_released_cube_and_park_exit(self):
        motion=Motion.__new__(Motion);motion.check_path=Mock();motion.plan_park=Mock()
        phase=lambda name:Phase(name,(dict(Q),dict(Q)))
        phases=tuple(phase(name) for name in ('approach','enter','grasp','extract_vertical','transfer','lower_to_support','release','retract'))
        plan=NS(phases=phases)
        with self.assertRaisesRegex(ValueError,'stationary released cube'):
            motion.preflight_cycle(plan,NS())
        plan.phases=phases[:-1]+(Phase('retract',(dict(Q),dict(Q)),placed_cube=NS()),)
        motion.preflight_cycle(plan,NS())
        motion.plan_park.assert_called_once()
        plan.phases=plan.phases[:-1]
        with self.assertRaisesRegex(ValueError,'retract'):
            motion.preflight_cycle(plan,NS())

    def test_park_cannot_bypass_latched_stop(self):
        rt=Runtime.__new__(Runtime);rt.stop_event=threading.Event();rt.stop_event.set()
        rt.robot=Mock();rt.cameras=Mock()
        with self.assertRaisesRegex(RuntimeError,'STOP'):
            rt.park_arm(dry=False,attended=True)
        rt.robot.assert_not_called()
        rt.robot.enable.assert_not_called()
        rt.robot.read_state.assert_not_called()

    def test_native_per_pair_queries_preserve_ordinary_collisions(self):
        robot=Robot('/home/jetsonnx/Projects/so101_autonomous_pick_place',ROOT/'runtime/unit')
        model=robot.backend.model
        ready=json.loads((ROOT/'runtime/contact-free-trial.json').read_text())['start']
        for q in (Q,ready):
            first=model.collides(model.pose(q));pairs=model.pair_collisions(q)
            self.assertEqual(bool(first),bool(pairs))
            if first:self.assertIn(first,pairs)
            self.assertEqual(len(model.pair_clearances(q)),19)

    def test_wrist_visibility_does_not_require_overview_identity(self):
        scene=Scene(NS())
        scene.raw={'wrist':[{'color':'green','quad':np.array([[0,0],[20,0],[20,20],[0,20]]),'center_px':[10,10]}]}
        public=scene.public()
        self.assertEqual(public['visible_wrist_count'],1)
        self.assertEqual(public['visible_overview_count'],0)
        self.assertIsNone(public['wrist_objects'][0]['associated_object_id'])
        self.assertFalse(public['wrist_objects'][0]['spatial_update'])

    def test_enable_preserves_an_existing_powered_hold(self):
        robot=Robot.__new__(Robot)
        robot.arm=NS(torque=True,torque_on=Mock())
        robot.owns_lease=Mock(return_value=True)
        robot.enable()
        robot.arm.torque_on.assert_not_called()
        robot.arm.torque=False
        robot.enable()
        robot.arm.torque_on.assert_called_once()

    def test_released_placement_guard_failure_latches_stop(self):
        rt=Runtime.__new__(Runtime)
        rt.check_cancel=Mock();rt.cameras=NS(check=Mock());rt.refresh=Mock()
        rt.robot=NS(first_command_at=None);rt.scene=NS(held=None)
        rt.report=Mock();rt.stop=Mock();rt.persist=Mock();rt.metrics={}
        def fail_after_command(attendance):
            rt.robot.first_command_at=time.monotonic()
            raise RuntimeError('retract pos: wrist_camera_board_lens: box rear wall')
        rt.resume_released_placement=fail_after_command
        rt.perform_prepare({'grasp_mode':'overhead_grasp','resume_released_placement':True})
        rt.stop.assert_called_once()
        self.assertTrue(rt.stop.call_args.kwargs['hardware_fault'])
        self.assertFalse(rt.results[0]['physical_verified'])

    def test_passive_current_image_does_not_invent_hidden_depth(self):
        record=json.loads((ROOT/'runtime/visual-measurement.json').read_text())
        feedback=json.loads((ROOT/'runtime/initial-feedback.json').read_text())['positions']
        robot=Robot('/home/jetsonnx/Projects/so101_autonomous_pick_place',ROOT/'runtime/unit')
        geometry=Geometry(robot.root)
        image=cv2.imread(str(ROOT/'runtime/visibility-current.jpg'))
        frame=Frame('overview',record['frame_id'],image,sum(record['capture_interval'])/2,
            record['capture_interval'][1],tuple(record['capture_interval']),np.asarray(record['joints_rad']),
            record['feedback_error_s'],feedback['gripper'],0.)
        report=geometry.measure_visible_attachment(frame,robot,json.loads((ROOT/'runtime/attachment-candidate.json').read_text()),now=frame.received_at)
        self.assertTrue(report['overview_fk_control']['passed'])
        self.assertGreater(report['visible_board_pixels'],100)
        self.assertIsNone(report['physical_gap_m'])
        self.assertFalse(report['cable_3d_pose_inferred'])
        self.assertFalse(report['motion_authorized'])
        self.assertGreaterEqual(report['visible_surface_clearance_bound']['lower_bound_m'],0.)
        with self.assertRaisesRegex(ValueError,'stale'):
            geometry.measure_visible_attachment(frame,robot,json.loads((ROOT/'runtime/attachment-candidate.json').read_text()),now=frame.received_at+2.)

    def test_pnp_geometry_identity_and_landmark_uncertainty(self):
        geometry=Geometry('/home/jetsonnx/Projects/so101_autonomous_pick_place')
        # Numerical projection identity, not a simulated or physically observed robot.
        points=np.array([[x,y,z] for x in (-.015,.015) for y in (-.012,.012) for z in (-.01,.01)])
        target=np.eye(4);target[:3,3]=[.26,0.,.04]
        pixels=geometry.project(points+target[:3,3])
        frame=Frame('overview','math-check',np.zeros((720,1280,3),np.uint8),time.time(),time.time(),(time.time(),time.time()))
        result=geometry.rigid_pose_from_landmarks(points,pixels,frame,None,.0001)
        self.assertLess(np.linalg.norm(np.asarray(result['base_from_object'])-target),1e-5)
        self.assertGreater(max(result['translation_sigma_m']),0.)
        with self.assertRaisesRegex(ValueError,'collinear'):
            geometry.rigid_pose_from_landmarks(np.column_stack((np.arange(4),np.zeros((4,2)))),np.zeros((4,2)),frame,None,.001)

    def test_repeated_fixed_camera_views_cannot_measure_depth(self):
        geometry=Geometry('/home/jetsonnx/Projects/so101_autonomous_pick_place')
        stamp=time.time();image=np.zeros((720,1280,3),np.uint8)
        observations=[{'frame':Frame('overview',str(i),image,stamp,stamp,(stamp,stamp)),'mask':np.ones((720,1280),np.uint8)} for i in range(3)]
        with self.assertRaisesRegex(ValueError,'no metric depth parallax'):
            geometry.reconstruct_feature_cloud(observations,None)

    def test_active_perception_cannot_move_to_resolve_its_own_startup_block(self):
        rt=Runtime.__new__(Runtime);rt.check_cancel=Mock();rt.scene=NS(held=None)
        rt.robot=NS(read_state=Mock(return_value={}),audit_state=Mock(return_value={'planning_valid':False}))
        rt.motion=Mock();rt.motion.driver=None
        with self.assertRaisesRegex(ValueError,'verified working area'):
            rt.refine_object('cube',None,None)
        rt.motion.execute_trajectory.assert_not_called()

    def test_active_perception_respects_force_retention_even_without_scene_attachment(self):
        rt=Runtime.__new__(Runtime);rt.check_cancel=Mock();rt.scene=NS(held=None)
        rt.robot=NS(read_state=Mock(),audit_state=Mock(return_value={'planning_valid':True}))
        rt.motion=Mock();rt.motion.driver=NS(retention=NS(armed=True))
        with self.assertRaisesRegex(ValueError,'empty jaw'):
            rt.refine_object('cube',None,None)
        rt.motion.execute_trajectory.assert_not_called()

    def test_one_placement_never_changes_calibration_or_establishes_bias(self):
        rt=Runtime.__new__(Runtime);rt.action_measurements=deque(maxlen=80);rt.metrics={}
        rt.robot=NS(read_state=Mock(return_value={'q':np.zeros(5),'gripper':20.}))
        rt.geometry=NS(fingerprint='fixture');rt.scene=NS(state=SceneState())
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            rt.output=Path(directory)
            for i in range(5):
                rt.record_action_measurement('placement',str(i),{'error_m':[.005,0.,0.],'position_sigma_m':[.001]*3})
                stats=rt.metrics['placement_residuals']
                self.assertFalse(stats['parameters_changed'])
                self.assertFalse(stats['hardware_limits_changed'])
                if i==0:self.assertFalse(stats['systematic_bias_candidate'])
            self.assertTrue(stats['systematic_bias_candidate'])
    def test_saved_native_box_observation_keeps_two_bodies_without_qualifying_depth(self):
        scan=json.loads((ROOT/'runtime/boxes-native-current.json').read_text())
        captured=scan['frame']['captured_at_unix_s']
        image=cv2.imread(str(Path(scan['artifacts'])/'frame.jpg'))
        wrist=cv2.imread(str(ROOT/'runtime/wrist-lit-8ms.jpg'))
        scene=Scene(Geometry('/home/jetsonnx/Projects/so101_autonomous_pick_place'))
        with patch('wristlab.scene.time.time',return_value=captured+1.):
            scene.install_box_proposals(scan)
            frames={name:Frame(name,'recorded:'+name,pixels,captured+1.,captured+1.,(captured+1.,captured+1.))
                    for name,pixels in (('overview',image),('wrist',wrist))}
            scene.update_scene(frames)
        self.assertEqual(len(scene.state.boxes),2)
        self.assertEqual(len(scene.state.objects),4)
        self.assertTrue(all(not box.evidence['metric_qualified'] for box in scene.state.boxes.values()))

    def test_runtime_states_distinguish_planning_execution_and_failure(self):
        rt = Runtime.__new__(Runtime)
        rt.blocking_conditions=Mock(return_value=[])
        rt.motion=NS(grasp_mode='wrist_refinement')
        rt.robot=NS(last_state=None,arm=None);rt.scene=NS(held=None,state=SceneState())
        rt.robot.motor_health={};rt.motor_warnings=[];rt.parked=False;rt.last_motion_unix=time.time()
        rt.park_error=rt.recovery_proposal=None;rt.physical_progress={}
        rt.initialized=False;rt.running=False;rt.phase='initializing';rt.detail='check'
        rt.stop_event=threading.Event();rt.pause_event=threading.Event();rt.fault_latched=False
        rt.mode=rt.started=rt.finished=rt.selected=None
        rt.metrics={};rt.log=[];rt.results=[]
        self.assertEqual(rt.status()['state'],'NOT_INITIALIZED')
        rt.initialized=True
        self.assertEqual(rt.status()['state'],'READY')
        rt.running=True;rt.phase='grasp_planning'
        self.assertEqual(rt.status()['state'],'PLANNING')
        rt.phase='moving'
        rt.blocking_conditions.return_value=[{'code':'SCENE_STALE','reason':'readiness observation stale during transfer'}]
        self.assertEqual(rt.status()['state'],'EXECUTING')
        rt.blocking_conditions.return_value=[];rt.phase='recovering'
        self.assertEqual(rt.status()['state'],'RECOVERING')
        rt.stop_event.set()
        self.assertEqual(rt.status()['state'],'SAFETY_BLOCKED')
        rt.stop_event.clear();rt.running=False;rt.phase='completed'
        self.assertEqual(rt.status()['state'],'COMPLETED')
        rt.phase='failed'
        self.assertEqual(rt.status()['state'],'FAILED')

    def test_native_auto_unfold_is_disabled_without_enabling_torque(self):
        robot = Robot.__new__(Robot)
        robot.arm=None;robot.acquire=Mock();robot.release=Mock();robot.output=ROOT/'runtime/unit'
        robot.nonce='recorded-test-owner'
        robot.backend=NS()
        robot.config=NS(robot={'port':'unused','calibration':'unused'})
        with patch('wristlab.robot.TaskArm') as native:
            robot.connect_once(lambda record:None)
        self.assertIs(native.return_value.unfold_first,False)
        native.return_value.raw.adopt_hold.assert_called_once()
        native.return_value.torque_on.assert_not_called()
        native.return_value.move_joints.assert_not_called()

    def test_restored_startup_component_matches_physically_used_source(self):
        previous=Path('/home/jetsonnx/Projects/so101_box_color_sorting_archived_20261106T085545/box_extract/mounted_arm.py')
        restored=ROOT/'box_extract/mounted_arm.py'
        self.assertEqual(hashlib.sha256(previous.read_bytes()).digest(),hashlib.sha256(restored.read_bytes()).digest())

    def test_planning_cache_never_applies_to_live_guard_checks(self):
        world = AccessoryWorld.__new__(AccessoryWorld)
        world.cache={};world.cache_enabled=True;world.metrics={'collision_checking_s':None}
        world.release_origin=None
        world._reason=Mock(return_value=None)
        world.reason(Q,target='recorded');world.reason(Q,target='recorded')
        self.assertEqual(world._reason.call_count,1)
        self.assertEqual(world.metrics['collision_cache_hits'],1)
        world.cache_enabled=False
        world._reason.return_value='new contact'
        self.assertEqual(world.reason(Q,target='recorded'),'new contact')
        self.assertEqual(world._reason.call_count,2)

    def test_intermediate_compensated_command_cannot_cross_obstacle(self):
        motion = Motion.__new__(Motion)
        motion.robot = NS(native_edge_samples=Robot.native_edge_samples,quantized_command=dict)
        motion.check_cancel = lambda:None
        motion.report = lambda *args:None
        motion.planner = NS(command_sag=3.7,_command_pose=lambda q:dict(q,shoulder_pan=q['shoulder_pan']+10))
        start,goal = dict(Q,shoulder_pan=0.),dict(Q,shoulder_pan=2.)
        world = NS(reason=lambda q,**policy:'obstacle' if 10.9<q['shoulder_pan']<11.1 else None)
        plan = ExtractionPlan('recorded','cube',(),np.zeros((8,3)),np.zeros(3),())
        with self.assertRaisesRegex(ValueError,'native command: obstacle'):
            motion.check_path(NS(phases=(Phase('transfer',(start,goal)),),target_id=plan.target_id,payload_local=plan.payload_local),world)
        with self.assertRaisesRegex(ValueError,'empty trajectory'):
            motion.check_path(NS(phases=(Phase('transfer',()),)),world)

    def test_table_slot_is_invalidated_when_an_object_occupies_it(self):
        scene = Scene(NS())
        polygon = np.array([[.35,-.06],[.47,-.06],[.47,.06],[.35,.06]])
        patch_data = SupportPatch(polygon,0.,.001,{'independent_metric_views':3,'free_surface_verified':True})
        scene.state.surfaces = [{'id':'table','patch':patch_data,'height_m':0.,'free_surface_verified':True}]
        self.assertEqual(len(scene.find_free_table_slots(np.full(3,SIDE))),1)
        scene.state.objects['occupied'] = object_at('occupied','red',.41,0.)
        self.assertEqual(scene.find_free_table_slots(np.full(3,SIDE)),[])

    def test_stack_height_and_top_first_disassembly(self):
        scene = Scene(NS())
        base = object_at('base','green',.26,0.)
        top = object_at('top','blue',.34,.1)
        scene.state.objects = {'base':base,'top':top}
        key,placement = experiments.stack_step(scene,2,'base')
        self.assertEqual(key,'top')
        self.assertAlmostEqual(placement.object_pose[2,3],1.5*SIDE)
        top.pose = placement.object_pose
        destination = Placement(np.eye(4),'table')
        with patch('wristlab.experiments.table_slot',return_value=destination):
            self.assertEqual(experiments.unstack_step(scene)[0],'top')

    def test_sequence_extracts_both_boxes_and_reobserves(self):
        scene = Scene(NS());scene.state.boxes = {'left':NS(),'right':NS()}
        refresh = Mock()
        with patch('wristlab.tasks.run_task',return_value=[]) as run:
            experiments.run_sequence(['empty_box'],scene,NS(),refresh,lambda:None)
        self.assertEqual([c.kwargs['destination'] for c in run.call_args_list],['left','right'])
        self.assertGreaterEqual(refresh.call_count,3)


if __name__=='__main__':
    unittest.main(verbosity=2)
