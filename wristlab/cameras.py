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
    def __init__(self, api, max_age=.5):
        self.api, self.max_age = api, max_age
        self.frames, self.history = {}, deque(maxlen=120)
        self.last_health = {}
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None
        self.on_native_stop = None

    def json(self, path, payload=None, timeout=2.):
        request = urllib.request.Request(self.api + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={} if payload is None else {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

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
                    self.on_native_stop()
                self.error = None
            except Exception as exc:
                self.error = str(exc)

    def latest_pair(self):
        return {name:self.get_latest_frame(name) for name in ('overview','wrist')}

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        self.latest_pair()

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
            frame = Frame(name, frame_id, image, (lo + hi) / 2., received, (lo, hi))
            state = self.get_robot_state_at(frame.timestamp)
            if state is not None:
                frame.joints, frame.feedback_error_s = state
            self.frames[name] = frame
        self.last_health = before
        return self.frames

    def record_state(self, timestamp, joints):
        self.history.append((float(timestamp), np.asarray(joints, float).copy()))

    def get_robot_state_at(self, timestamp):
        pairs = list(zip(self.history, list(self.history)[1:]))
        match = next(((a, b) for a, b in reversed(pairs) if a[0] <= timestamp <= b[0] and b[0]-a[0] <= .1), None)
        if match is None:
            return None
        a, b = match
        alpha = (timestamp-a[0]) / max(1e-9, b[0]-a[0])
        return a[1] + alpha*(b[1]-a[1]), max(timestamp-a[0], b[0]-timestamp)

    def get_latest_frame(self, camera_id):
        frame = self.frames[camera_id]
        if time.time() - frame.capture_interval[0] > self.max_age:
            raise RuntimeError(camera_id + ': stale frame')
        state = self.get_robot_state_at(frame.timestamp)
        if state is not None:
            frame.joints, frame.feedback_error_s = state
        return frame
