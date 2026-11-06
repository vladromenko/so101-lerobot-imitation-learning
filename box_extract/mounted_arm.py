"""Camera-aware lifecycle using the existing C controller; no blind folded home."""
from so101_pick_place.hardware.task_arm import TaskArm
from so101_pick_place.experiments.box_extraction.planner import ExtractionPlanner
from so101_pick_place.experiments.box_extraction.search import dense_edge
from .planning import TunedExtractionPlanner

JOINTS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')


def ready_path(backend,initial,ready,world,predicate):
    if world.reason(initial) is None:
        planner=TunedExtractionPlanner(backend)
        intermediate=dict(ready,gripper=initial['gripper'])
        path,_=planner.transport(initial,intermediate,world)
        return tuple(path)+tuple(planner.edge(intermediate,ready,world))[1:]
    return validated_segments(initial,[{'wrist_flex':0.},{'wrist_roll':0.,'gripper':35.},ready],predicate)


def validated_segments(start,goals,predicate):
    path=[dict(start)];previous=dict(start)
    for goal in goals:
        final=dict(previous)|dict(goal)
        def valid(values):
            return predicate(dict(zip(JOINTS,map(float,values)))) is None
        points=dense_edge([previous[j] for j in JOINTS],[final[j] for j in JOINTS],valid,step_deg=.5)
        if points is None:
            raise ValueError('camera-aware return/unfold segment is not collision free')
        path.extend(dict(zip(JOINTS,map(float,p))) for p in points[1:]);previous=final
    return tuple(path)


class MountedTaskArm(TaskArm):
    unfold_first=False

    def __init__(self,*args,mount_backend,**kwargs):
        self.mount_backend=mount_backend
        self.parking_target=None
        self.parking_world=None
        self.external_check=None
        super().__init__(*args,**kwargs)

    def follow_checked(self,path,step):
        path=tuple(path)
        travel=sum(max(abs(b[j]-a[j]) for j in JOINTS) for a,b in zip(path,path[1:]))
        self.follow(path,duration=max(1.,travel/6.),settle=.5,within=.7,lead=False,step=step)

    def prepare_ready(self,ready,predicate,world_provider):
        self.parking_target=dict(ready)
        self.external_check=predicate
        self.parking_world=world_provider
        initial=self.read()[0]
        path=ready_path(self.mount_backend,initial,ready,world_provider(),predicate)
        for q in path:
            reason=predicate(self.mount_backend.driver_command_goal(q,self.raw.sag))
            if reason:raise RuntimeError('ready command: '+reason)
        self.follow_checked(path,'camera_aware_unfold')

    def home(self):
        if self.parking_target is None or self.parking_world is None:
            self.stop()
            raise RuntimeError('no validated camera-aware parking target; holding, native fold-home disabled')
        current=self.read()[0];world=self.parking_world()
        self.parking_target=dict(self.parking_target,gripper=current['gripper'])
        path,_=TunedExtractionPlanner(self.mount_backend).transport(current,self.parking_target,world)
        for joints in path:
            reason=self.external_check(joints)
            if reason:
                raise RuntimeError('camera-aware parking refused: '+reason)
        self.follow_checked(path,'camera_aware_parking')
        reason=self.external_check(self.read()[0])
        if reason:raise RuntimeError('camera-aware parking endpoint: '+reason)
        self.stop()
        self.log('PARKED_CAMERA_CLEAR: ready posture held; no automatic wrist fold or torque-off')


def task_arm(backend,**kwargs):
    if hasattr(backend.model,'accessory_profile'):
        return MountedTaskArm(mount_backend=backend,**kwargs)
    return TaskArm(**kwargs)
