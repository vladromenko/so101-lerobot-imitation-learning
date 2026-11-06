"""Bounded recovery: observe changed scenes; never repeat a blind contact."""


class Recovery:
    def __init__(self, max_attempts=2):
        self.max_attempts = max_attempts
        self.failures = {}

    def recover(self, object_id, reason, scene, motion_started, holding):
        signature = (object_id,reason)
        attempts = self.failures.get(signature,0)+1
        self.failures[signature] = attempts
        if holding or motion_started:
            return {'retry':False,'action':'hold','reason':reason}
        if attempts >= self.max_attempts:
            return {'retry':False,'action':'defer','reason':reason}
        return {'retry':True,'action':'observe_changed_scene','reason':reason,
                'requires_new_geometry':True}
