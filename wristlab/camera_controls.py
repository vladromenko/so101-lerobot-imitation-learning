"""V4L2 control ioctls only: the native service remains the sole stream owner."""
import fcntl
import os
import struct

CONTROLS = {'auto_exposure':0x009a0901,'exposure_time_absolute':0x009a0902,
            'gain':0x00980913,'power_line_frequency':0x00980918,'exposure_dynamic_framerate':0x009a0903}
G_CTRL,S_CTRL = 0xc008561b,0xc008561c


def read_controls(device):
    fd = os.open(device,os.O_RDWR|os.O_NONBLOCK)
    try:
        result = {}
        for name,cid in CONTROLS.items():
            value = bytearray(struct.pack('Ii',cid,0))
            fcntl.ioctl(fd,G_CTRL,value,True)
            result[name] = struct.unpack('Ii',value)[1]
        return result
    finally:
        os.close(fd)


def apply_controls(device, settings):
    before = read_controls(device)
    fd = os.open(device,os.O_RDWR|os.O_NONBLOCK)
    try:
        for name,value in settings.items():
            if name not in CONTROLS:
                raise ValueError('unsupported V4L2 control: '+name)
            fcntl.ioctl(fd,S_CTRL,struct.pack('Ii',CONTROLS[name],int(value)))
    finally:
        os.close(fd)
    after = read_controls(device)
    if any(after[name] != int(value) for name,value in settings.items()):
        raise RuntimeError('camera did not accept requested controls')
    return {'before':before,'after':after,'opened_video_stream':False}
