"""Control-contract checks on recorded feedback; no bus, cameras or robot simulation.

Run with the project's hardware Python. Actuator calls are recorded, never sent.
This verifies software failure handling, not physical manipulation.
"""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wristlab.actions import Actions
from wristlab.motion import Motion
from wristlab.robot import Robot
from wristlab.scene import Scene
from wristlab.recovery import Recovery
from wristlab import tasks,experiments
from wristlab.types import ObjectState,Placement,SceneState,ActionResult
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
    scene.native_object = Mock(return_value=cube)
    state = {'positions':dict(Q),'gripper':Q['gripper']}
    robot = NS(read_state=Mock(return_value=state),arm=NS(read=Mock(return_value=(dict(Q),{},{}))),
        backend=NS(model=NS(frames=lambda q:{'gripper_frame_link':np.eye(4)})))
    local = vertices-vertices.mean(0)
    def phase(name):
        return Phase(name,(dict(Q),dict(Q)),name in ('extract_vertical','release'))
    plan = ExtractionPlan('recorded',obj.id,tuple(phase(s) for s in ('open','approach','enter','grasp')),
                          local,np.array([.3,0.,SIDE/2]),())
    rest = ExtractionPlan('recorded',obj.id,tuple(phase(s) for s in ('extract_vertical','transfer','release','retract')),
                          local,np.array([.3,0.,SIDE/2]),())
    driver = NS(confirm_grasp=Mock(return_value=local),release_confirmed=Mock(),retention=NS(armed=True))
    motion = NS(qualification_reasons=Mock(return_value=[]),make_world=Mock(return_value=NS()),
        check_path=Mock(),contact_free_passed=True,driver=driver,planner=NS(),execute_trajectory=Mock(),
        replan_held=Mock(return_value=(rest,NS())),return_to_start=Mock())
    rt = NS(scene=scene,robot=robot,motion=motion,metrics={},report=Mock(),refresh=Mock(),
        check_cancel=Mock(),wait_pause=Mock(),ensure_driver=Mock(),stop=Mock(),recovery=Recovery(),
        mode='relocate',physical_relocate_count=0,cameras=NS())
    action = Actions(rt)
    action.capture_grip_view = Mock(return_value={'frame_id':'anchor'})
    action.verify_pick = Mock(return_value=True)
    action.verify_placement = Mock(return_value=(True,obj.pose.copy()))
    placement = Placement(obj.pose.copy(),'box-1')
    alignment = {'observed_top_m':(obj.pose[:3,3]+[0.,0.,SIDE/2]).tolist()}
    return obj,rt,action,placement,plan,alignment


class RelocateContract(unittest.TestCase):
    def invoke(self, setup, **options):
        obj,rt,action,placement,plan,alignment = setup
        with patch('wristlab.actions.choose_grasp',return_value=(None,plan,NS())),patch('wristlab.actions.visual_alignment',return_value=alignment):
            result = action.relocate(obj.id,placement,**options)
        return result

    def test_full_contract_removes_static_payload_and_returns_using_observed_cube(self):
        setup = action_contract();obj,rt,action,*_ = setup
        result = self.invoke(setup)
        self.assertTrue(result.success,result.reason)
        self.assertNotIn(obj.id,rt.motion.replan_held.call_args.args[0].cubes)
        names = [c.args[0].name for c in rt.motion.execute_trajectory.call_args_list]
        self.assertLess(names.index('extract_vertical'),names.index('release'))
        action.verify_pick.assert_called_once()
        action.verify_placement.assert_called_once()
        rt.motion.return_to_start.assert_called_once()
        self.assertIs(rt.motion.return_to_start.call_args.args[-1],rt.scene.native_object.return_value)
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
        rt.ensure_driver.assert_not_called()
        rt.motion.execute_trajectory.assert_not_called()
        rt.motion.driver.confirm_grasp.assert_not_called()


class GeometryPolicy(unittest.TestCase):
    def test_intermediate_compensated_command_cannot_cross_obstacle(self):
        motion = Motion.__new__(Motion)
        motion.robot = NS(native_edge_samples=Robot.native_edge_samples)
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
