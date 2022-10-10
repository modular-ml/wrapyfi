import io
import json
import base64

from wrapyfi.utils import *

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@PluginRegistrar.register
class PytorchTensor(Plugin):
    def __init__(self, load_torch_device=None, map_torch_devices=None, **kwargs):
        self.map_torch_devices = map_torch_devices or {}
        if load_torch_device is not None:
            self.map_torch_devices['default'] = load_torch_device

    def encode(self, obj, *args, **kwargs):
        if HAVE_TORCH and isinstance(obj, torch.Tensor):
            with io.BytesIO() as memfile:
                torch.save(obj, memfile)
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            obj_device = self.map_torch_devices.get(str(obj.device), self.map_torch_devices.get('default', None))
            if obj_device is None:
                obj_device = str(obj.device)
            return True, dict(__wrapyfi__=('torch.Tensor', obj_data, obj_device))
        else:
            return False, None

    def decode(self, obj_type, obj_full, *args, **kwargs):
        if HAVE_TORCH and obj_type == 'torch.Tensor':
            with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
                return True, torch.load(memfile, map_location=obj_full[2])
        else:
            return False, None

