import io
import base64
from functools import lru_cache

import numpy as np

from wrapyfi.utils import *


try:
    import mxnet
    HAVE_MXNET = True
except ImportError:
    HAVE_MXNET = False


def mxnet_device_to_str(device):
    if device is None:
        return 'cpu:0'
    elif isinstance(device, dict):
        device_rets = {}
        for k, v in device.items():
            device_rets[mxnet_device_to_str(k)] = mxnet_device_to_str(v)
        return device_rets
    elif isinstance(device, mxnet.Context):
        return f'{device.device_type.replace("gpu", "cuda")}:{device.device_id}'
    elif isinstance(device, str):
        return device.replace("gpu", "cuda")
    else:
        raise ValueError(f'Unknown device type {type(device)}')


@lru_cache(maxsize=None)
def mxnet_str_to_device(device):
    if device is None:
        return mxnet.cpu()
    elif isinstance(device, mxnet.Context):
        return device
    elif isinstance(device, str):
        try:
            device_type, device_id = device.split(':')
        except ValueError:
            device_type = device
            device_id = 0
        return mxnet.Context(device_type.replace("cuda", "gpu"), int(device_id))
    else:
        raise ValueError(f'Unknown device type {type(device)}')


@PluginRegistrar.register(types=None if not HAVE_MXNET else mxnet.nd.NDArray.__mro__[:-1])
class MXNetTensor(Plugin):
    def __init__(self, load_mxnet_device=None, map_mxnet_devices=None, **kwargs):
        self.map_mxnet_devices = map_mxnet_devices or {}
        if load_mxnet_device is not None:
            self.map_mxnet_devices['default'] = load_mxnet_device
        self.map_mxnet_devices = mxnet_device_to_str(self.map_mxnet_devices)

    def encode(self, obj, *args, **kwargs):
        with io.BytesIO() as memfile:
            np.save(memfile, obj.asnumpy())
            obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
        obj_device = mxnet_device_to_str(obj.context)
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_device))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
            obj_device = self.map_mxnet_devices.get(obj_full[2], self.map_mxnet_devices.get('default', None))
            if obj_device is not None:
                obj_device = mxnet_str_to_device(obj_device)
                return True, mxnet.nd.array(np.load(memfile), ctx=obj_device)
            else:
                return True, mxnet.nd.array(np.load(memfile))
