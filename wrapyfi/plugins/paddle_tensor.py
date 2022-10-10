import io
import json
import base64

from wrapyfi.utils import *

try:
    import paddle
    HAVE_PADDLE = True
except ImportError:
    HAVE_PADDLE = False


@PluginRegistrar.register
class PaddleTensor(Plugin):
    def __init__(self, load_paddle_device=None, map_paddle_devices=None, **kwargs):
        self.map_paddle_devices = map_paddle_devices or {}
        if load_paddle_device is not None:
            self.map_paddle_devices['default'] = load_paddle_device

    def encode(self, obj, *args, **kwargs):
        if HAVE_PADDLE and isinstance(obj, paddle.Tensor):
            with io.BytesIO() as memfile:
                paddle.save(obj, memfile)
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            obj_device = obj.get_device()
            return True, dict(__wrapyfi__=('paddle.Tensor', obj_data, obj_device))
        else:
            return False, None

    def decode(self, obj_type, obj_full, *args, **kwargs):
        if HAVE_PADDLE and obj_type == 'paddle.Tensor':
            with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
                obj_device = self.map_paddle_devices.get(obj_full[2], self.map_paddle_devices.get('default', None))
                return True, paddle.load(memfile).set_device(obj_device)
        else:
            return False, None
