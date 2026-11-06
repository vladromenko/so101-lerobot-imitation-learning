"""Single native driver owner. Native degrees exist only at this boundary."""
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
from so101_pick_place.config import load_config
from so101_pick_place.kinematics import So101ModelPlanner
from so101_pick_place.kinematics.so101_backend import ARM_JOINTS
from box_extract.mounted_arm import MountedTaskArm as TaskArm
from so101_pick_place.hardware.robot import ReadOnlySO101


class Robot:
    def __init__(self, native_root, output):
        self.root, self.output = Path(native_root), Path(output)
        self.config = load_config(self.root / 'config/default.json')
        robot = self.config.robot
        self.backend = So101ModelPlanner(robot['calibration'], offsets=robot.get('kinematic_offsets_deg'))
        self.arm = None
        self.lease = self.root / 'runtime/single_cube_executor.json'
        self.nonce = None
        self.last_state = None
        self.on_tick = None
        self.first_command_at = None
        self.audit_cache = None

    @staticmethod
    def from_native(joints):
        return np.deg2rad([joints[name] for name in ARM_JOINTS])

    @staticmethod
    def to_native(q, gripper):
        return dict(zip(ARM_JOINTS, np.rad2deg(q).tolist())) | {'gripper': float(gripper)}

    @staticmethod
    def native_edge_samples(start, goal):
        # Native phases carry degrees and the native jaw percentage. Keep their
        # interpolation at the driver boundary; new task geometry stays in SI.
        span = max(abs(float(goal[j])-float(start[j])) for j in start)
        for alpha in np.linspace(0.,1.,max(2,int(np.ceil(span/.5))+1)):
            yield {j:float(start[j]+alpha*(goal[j]-start[j])) for j in start}

    def acquire(self, kind):
        nonce = uuid.uuid4().hex
        fd = os.open(self.lease, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump({'pid': os.getpid(), 'script': 'wristlab.ui', 'owner_nonce': nonce, 'kind': kind,
                       'artifacts': str(self.output)}, handle)
        self.nonce = nonce

    def owns_lease(self):
        try:
            return self.nonce is not None and json.loads(self.lease.read_text()).get('owner_nonce') == self.nonce
        except (OSError, ValueError):
            return False

    def release(self):
        if self.owns_lease():
            self.lease.unlink()
        self.nonce = None

    def read_state(self):
        if self.arm is not None:
            if not self.owns_lease():
                raise RuntimeError('hardware lease was lost; no further bus reads')
            q, loads, currents = self.arm.read()
            state = {'positions': q, 'loads': loads, 'currents': currents}
        else:
            self.acquire('read_only_feedback')
            try:
                state = ReadOnlySO101(self.config.robot['port'], self.config.robot['calibration']).read()
            finally:
                self.release()
        state['timestamp'] = time.time()
        state['q'] = self.from_native(state['positions'])
        state['gripper'] = float(state['positions']['gripper'])
        self.last_state = state
        if self.owns_lease():
            record = json.loads(self.lease.read_text())
            record.update(robot_joint_deg=state['positions'],feedback_time_unix_s=state['timestamp'],state='MONITORING' if not self.arm.torque else 'HOLDING')
            temporary = self.lease.with_name('.wristlab-feedback-'+str(os.getpid())+'.tmp')
            temporary.write_text(json.dumps(record))
            temporary.replace(self.lease)
        return state

    def audit_state(self, state):
        from .diagnostics import audit_pose
        model = self.backend.model
        key = (tuple(state['positions'][j] for j in (*ARM_JOINTS,'gripper')),
               model.margin,model.gap,model.table_z,tuple(model.offsets),str(model.ranges))
        if self.audit_cache is None or self.audit_cache[0]!=key:
            self.audit_cache = key,audit_pose(self,state)
        data = self.audit_cache[1] | {'timestamp':state['timestamp']}
        return data | {'valid':data['planning_valid'],'problems':data['planning_margin_problems']}

    def forward_kinematics(self, q):
        return self.link_pose(q, 'gripper_frame_link')

    def link_pose(self, q, name):
        return np.asarray(self.backend.model.frames(self.to_native(q, 35.))[name])

    def solve_ik(self, target_pose, q_seed):
        from scipy.optimize import least_squares
        target = np.asarray(target_pose)
        limits = np.deg2rad(np.asarray(self.backend.model.ranges))
        def residual(q):
            actual = self.forward_kinematics(q)
            return np.r_[1000.*(actual[:3,3]-target[:3,3]), 30.*(actual[:3,0]-target[:3,0])]
        fit = least_squares(residual, q_seed, bounds=(limits[:,0],limits[:,1]), max_nfev=100)
        if np.linalg.norm(residual(fit.x)[:3]) > 1.5:
            raise ValueError('no feasible SO-101 position and closing-axis IK')
        native = self.to_native(fit.x, 35.)
        pose = self.backend.model.pose(native)
        if self.backend.model.problems(pose) or self.backend.model.collides(pose):
            raise ValueError('IK violates native limits or self collision')
        return fit.x

    def connect_once(self, tick):
        if self.arm is None:
            self.acquire('wristlab_executor')
            try:
                r = self.config.robot
                self.arm = TaskArm(mount_backend=self.backend,port=r['port'], calibration_file=r['calibration'], offsets=r.get('kinematic_offsets_deg'),
                    on_tick=tick, hold_on_success=True, out=self.output, name='wristlab-'+self.nonce[:12])
                # Arm._ready() otherwise silently invokes the stock unfold
                # before move_joints/follow, outside our checked trajectory.
                self.arm.unfold_first = False
                self.arm._echo = False
                self.arm.check()
            except BaseException:
                if self.arm is not None:
                    self.arm.close()
                self.arm = None
                self.release()
                raise
        return self.arm

    def enable(self):
        if self.arm is None or not self.owns_lease():
            raise RuntimeError('native driver is not the exclusive bus owner')
        self.arm.torque_on()

    def send_joint_targets(self, q, gripper):
        if self.arm is None or not self.owns_lease():
            raise RuntimeError('no exclusive native driver')
        if self.first_command_at is None:
            self.first_command_at = time.monotonic()
        self.arm.move_joints(self.to_native(q, gripper), speed=5., check=True, step='wristlab_checked')

    def stop_motion(self):
        if self.arm is not None:
            self.arm.stop()

    def close(self):
        if self.arm is not None:
            self.arm.stop()
            self.arm.close()
            self.arm = None
        self.release()
