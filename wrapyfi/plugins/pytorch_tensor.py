import io
import json
import base64

from wrapyfi.utils import *

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@PluginRegistrar.register(types=None if not HAVE_TORCH else torch.Tensor.__mro__[:-1])
class PytorchTensor(Plugin):
    def __init__(self, load_torch_device=None, map_torch_devices=None, **kwargs):
        self.map_torch_devices = map_torch_devices or {}
        if load_torch_device is not None:
            self.map_torch_devices['default'] = load_torch_device

    def encode(self, obj, *args, **kwargs):
        with io.BytesIO() as memfile:
            torch.save(obj, memfile)
            obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
        obj_device = str(obj.device)
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_device))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
            obj_device = self.map_torch_devices.get(obj_full[2], self.map_torch_devices.get('default', None))
            if obj_device is not None:
                return True, torch.load(memfile, map_location=obj_device)
            else:
                return True, torch.load(memfile)

