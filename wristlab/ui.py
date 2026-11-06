"""Integrate the local runtime into the original work/tutorial panel on 8031."""
import os
import fcntl
from pathlib import Path
import signal
import threading
import time
import cv2
from fastapi import HTTPException
from fastapi.responses import Response,JSONResponse
from box_extract.ui.server import app
from .runtime import Runtime

root = Path(__file__).resolve().parents[1]
runtime = Runtime(root)
replaced = {'/api/run','/api/stop','/api/status','/api/scene','/snapshot.jpg','/api/live'}
app.router.routes[:] = [route for route in app.router.routes if getattr(route,'path','') not in replaced]


@app.post('/api/run')
def run(payload: dict):
    try:
        runtime.start(payload)
        return {'started':True}
    except (ValueError,RuntimeError) as exc:
        return JSONResponse(
            status_code=409,content={'detail':str(exc),'state':runtime.status()['state'],
                                    'blocking_conditions':runtime.blocking_conditions()})


@app.post('/api/task-plan')
def task_plan(payload: dict):
    try:
        return runtime.preview(payload)
    except (ValueError,KeyError) as exc:
        raise HTTPException(409,str(exc)) from exc


@app.get('/api/diagnostics')
def diagnostics():
    import json
    path = root/'runtime/escape-proposal.json'
    return {'current':None if runtime.robot.last_state is None else runtime.robot.audit_state(runtime.robot.last_state),
            'visual_measurements':runtime.scene.passive_measurements,
            'escape_proposal':None if not path.exists() else json.loads(path.read_text()),
            'robot_motion_commanded':runtime.robot.first_command_at is not None}


@app.post('/api/stop')
def stop():
    return runtime.stop()


@app.post('/api/pause')
def pause(payload: dict):
    if payload.get('paused',True):
        runtime.pause_event.set()
    else:
        runtime.pause_event.clear()
    return {'paused':runtime.pause_event.is_set(),'takes_effect':'next guarded phase boundary'}


@app.post('/api/reset')
def reset():
    try:
        runtime.reset()
        return {'reset_queued':True,'motion_started':False}
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@app.post('/api/prepare')
def prepare(payload: dict):
    try:
        runtime.prepare(payload)
        return {'queued':True,'scope':'attended overhead geometry' if payload.get('grasp_mode')=='overhead_grasp' else 'attended folded exit only'}
    except (ValueError,RuntimeError) as exc:
        raise HTTPException(409,str(exc)) from exc


@app.get('/api/status')
def status():
    return runtime.status()


@app.get('/api/scene')
def scene():
    return runtime.scene_public()


@app.post('/api/select')
def select(payload: dict):
    try:
        return runtime.select(payload)
    except (KeyError,ValueError) as exc:
        raise HTTPException(409,str(exc)) from exc


@app.get('/api/live')
def live():
    state = runtime.robot.last_state
    return {'time':None if state is None else state['timestamp'],
            'age_s':None if state is None else time.time()-state['timestamp'],
            'pos':None if state is None else state['positions'],'source':'wristlab native encoder read'}


@app.get('/snapshot.jpg')
def overview():
    return snapshot('overview')


@app.get('/snapshot/{name}.jpg')
def snapshot(name: str):
    try:
        frame = runtime.cameras.get_latest_frame(name)
        ok,data = cv2.imencode('.jpg',frame.image,[cv2.IMWRITE_JPEG_QUALITY,90])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        runtime.frame_history.append({name:frame})
        return Response(data.tobytes(),media_type='image/jpeg',headers={
            'Cache-Control':'no-store','X-Frame-Id':frame.frame_id,'X-Captured-At':str(frame.timestamp),
            'X-Capture-Interval':','.join(map(str,frame.capture_interval))})
    except (KeyError,RuntimeError) as exc:
        raise HTTPException(503,str(exc)) from exc


def main():
    import uvicorn
    lock = (root/'runtime/wristlab-service.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Wristlab service is already running')
    server = uvicorn.Server(uvicorn.Config(app,host='0.0.0.0',port=int(os.environ.get('PORT','8031')),log_level='warning'))
    thread = threading.Thread(target=server.run,name='wristlab-http',daemon=True)
    thread.start()
    def stop_signal(*args):
        runtime.stop(forward=runtime.robot.arm is not None or runtime.running)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    try:
        runtime.loop()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.robot.close()
        runtime.cameras.close()
        server.should_exit = True
        thread.join(timeout=2.)


if __name__ == '__main__':
    main()
