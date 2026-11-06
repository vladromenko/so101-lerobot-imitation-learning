"""Read the already-open native cameras. Never opens a video device."""
from collections import deque
import json
import time
import threading
import urllib.request
import cv2
import numpy as np
from .types import Frame


class Cameras:
    def __init__(self, api, max_age=.5, controls=None):
        self.api, self.max_age = api, max_age
        self.controls = controls or {}
        self.exposure_s = {name:values.get('exposure_time_absolute',0)/10000. for name,values in self.controls.items()}
        self.frames, self.history = {}, deque(maxlen=120)
        self.last_health = {}
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None
        self.on_native_stop = None
        self.on_frames = None
        self.frame_ready = threading.Condition()
        self.cache_refreshes = deque(maxlen=20)

    def json(self, path, payload=None, timeout=2.):
        request = urllib.request.Request(self.api + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={} if payload is None else {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def apply_controls(self):
        from .camera_controls import apply_controls
        health = self.json('/api/camera-health')['cameras']
        return {name:apply_controls(health[name]['device'],values) for name,values in self.controls.items()}

    def start_capture(self):
        try:
            self.refresh()
        finally:
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop,name='native-camera-reader',daemon=True)
                self.thread.start()
        return self.frames

    def _loop(self):
        while not self.stop_event.wait(.15):
            try:
                self.refresh()
                from .vision import frame_quality
                quality = {name:frame_quality(frame,40. if name=='overview' else 15.) for name,frame in self.frames.items()}
                invalid = [name+': '+q['reason'] for name,q in quality.items() if not q['usable']]
                if invalid:
                    raise RuntimeError('; '.join(invalid))
                native = self.json('/api/status')
                if native.get('state') == 'EMERGENCY_STOP' and self.on_native_stop is not None:
                    self.on_native_stop(native.get('updated_at'))
                self.error = None
            except Exception as exc:
                self.error = str(exc)
            if self.error is None and self.on_frames is not None:
                self.on_frames()

    def latest_pair(self):
        return {name:self.get_latest_frame(name) for name in ('overview','wrist')}

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        try:
            self.latest_pair()
        except RuntimeError as exc:
            if ': stale frame' not in str(exc):
                raise
            began = time.time()
            old = {name:began-frame.capture_interval[0] for name,frame in self.frames.items()}
            # Hold the command callback until actual fresh native snapshots
            # arrive. Neither the age bound nor a real camera failure is waived.
            self.refresh()
            self.latest_pair()
            self.cache_refreshes.append({'at':began,'old_age_s':old,'fetch_s':time.time()-began,
                'native_age_s':{name:value['age_s'] for name,value in self.last_health.items()}})

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)

    def refresh(self):
        before = self.json('/api/camera-health')['cameras']
        for name in ('overview', 'wrist'):
            health = before[name]
            if not health.get('available') or health.get('age_s', 99) > self.max_age:
                raise RuntimeError(name + ': camera missing or stale')
            with urllib.request.urlopen(self.api + '/snapshot/' + name + '.jpg', timeout=1.) as response:
                data = response.read()
            received = time.time()
            after = self.json('/api/camera-health')['cameras'][name]
            lo, hi = float(health['captured_at_unix_s']), float(after['captured_at_unix_s'])
            if hi < lo or received - lo > self.max_age:
                raise RuntimeError(name + ': snapshot timing is not bounded')
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(name + ': invalid JPEG')
            # Snapshot API supplies no sequence header. Preserve the actual interval,
            # rather than claiming an exact sequence from a different HTTP response.
            frame_id = f"{name}:{health['sequence']}-{after['sequence']}"
            exposure = self.exposure_s.get(name,0.)
            frame = Frame(name,frame_id,image,(lo+hi-exposure)/2.,received,(lo-exposure,hi))
            state = self.get_robot_state_at(frame.timestamp)
            if state is not None:
                frame.joints, frame.feedback_error_s, frame.gripper = state
            with self.frame_ready:
                previous = self.frames.get(name)
                if previous is None or previous.timestamp<=frame.timestamp:
                    self.frames[name] = frame
        self.last_health = before
        with self.frame_ready:
            self.frame_ready.notify_all()
        return self.frames

    def wait_for_frame(self, camera_id, after_time, check_cancel, timeout=2.):
        deadline = time.monotonic()+timeout
        with self.frame_ready:
            frame = self.frames.get(camera_id)
            while frame is None or frame.capture_interval[0] < after_time:
                check_cancel()
                remaining = deadline-time.monotonic()
                if remaining<=0:
                    raise RuntimeError('camera did not supply a complete post-motion exposure')
                self.frame_ready.wait(min(.1,remaining))
                frame = self.frames.get(camera_id)
        return self.get_latest_frame(camera_id)

    def record_state(self, timestamp, joints, gripper=None):
        self.history.append((float(timestamp),np.asarray(joints,float).copy(),gripper))

    def get_robot_state_at(self, timestamp):
        history = list(self.history)
        pairs = list(zip(history,history[1:]))
        match = next(((a, b) for a, b in reversed(pairs) if a[0] <= timestamp <= b[0] and (b[0]-a[0] <= .1 or (b[0]-a[0] <= .5 and np.max(np.abs(b[1]-a[1])) < .002))), None)
        if match is None:
            return None
        a, b = match
        alpha = (timestamp-a[0]) / max(1e-9, b[0]-a[0])
        gripper = None if a[2] is None or b[2] is None else a[2]+alpha*(b[2]-a[2])
        return a[1]+alpha*(b[1]-a[1]),max(timestamp-a[0],b[0]-timestamp),gripper

    def get_latest_frame(self, camera_id):
        frame = self.frames[camera_id]
        if time.time() - frame.capture_interval[0] > self.max_age:
            raise RuntimeError(camera_id + ': stale frame')
        state = self.get_robot_state_at(frame.timestamp)
        if state is not None:
            frame.joints, frame.feedback_error_s, frame.gripper = state
        history = list(self.history)
        before = [sample for sample in history if sample[0] <= frame.capture_interval[0]]
        after = [sample for sample in history if sample[0] >= frame.capture_interval[1]]
        if before and after:
            samples = [before[-1],*[sample for sample in history if frame.capture_interval[0] < sample[0] < frame.capture_interval[1]],after[0]]
            frame.motion_during_exposure_rad = float(np.ptp(np.array([sample[1] for sample in samples]),axis=0).max())
        return frame
