import io
import base64
from functools import lru_cache

import numpy as np

from wrapyfi.utils import *


try:
    import paddle
    HAVE_PADDLE = True
except ImportError:
    HAVE_PADDLE = False


def paddle_device_to_str(device):
    if device is None:
        return paddle.get_device()
    elif isinstance(device, dict):
        device_rets = {}
        for k, v in device.items():
            device_rets[paddle_device_to_str(k)] = paddle_device_to_str(v)
        return device_rets
    elif isinstance(device, paddle.fluid.libpaddle.Place):
        return str(device)[6:-1].replace("cuda", "gpu")
    elif isinstance(device, str):
        return device.replace("cuda", "gpu")
    else:
        raise ValueError(f'Unknown device type {type(device)}')


@lru_cache(maxsize=None)
def paddle_str_to_device(device):
    if device is None:
        return paddle.get_device()
    elif isinstance(device, paddle.fluid.libpaddle.Place):
        return device
    elif isinstance(device, str):
        try:
            device_type, device_id = device.split(':')
        except ValueError:
            device_type = device
        return paddle.device._convert_to_place(device_type.replace("cuda", "gpu"))
    else:
        raise ValueError(f'Unknown device type {type(device)}')


@PluginRegistrar.register
class PaddleTensor(Plugin):
    def __init__(self, load_paddle_device=None, map_paddle_devices=None, **kwargs):
        self.map_paddle_devices = map_paddle_devices or {}
        if load_paddle_device is not None:
            self.map_paddle_devices['default'] = load_paddle_device
        self.map_paddle_devices = paddle_device_to_str(self.map_paddle_devices)

    def encode(self, obj, *args, **kwargs):
        if HAVE_PADDLE and isinstance(obj, paddle.Tensor):
            with io.BytesIO() as memfile:
                paddle.save(obj, memfile)
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            obj_device = paddle_device_to_str(obj.place)
            return True, dict(__wrapyfi__=('paddle.Tensor', obj_data, obj_device))
        else:
            return False, None

    def decode(self, obj_type, obj_full, *args, **kwargs):
        if HAVE_PADDLE and obj_type == 'paddle.Tensor':
            with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
                obj_device = self.map_paddle_devices.get(obj_full[2], self.map_paddle_devices.get('default', None))
                if obj_device is not None:
                    obj_device = paddle_str_to_device(obj_device)
                    return True, paddle.Tensor(paddle.load(memfile), place=obj_device)
                else:
                    return True, paddle.load(memfile)
        else:
            return False, None

