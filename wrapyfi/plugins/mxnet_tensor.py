import io
import base64

import numpy as np

from wrapyfi.utils import *


try:
    import mxnet
    HAVE_MXNET = True
except ImportError:
    HAVE_MXNET = False


@PluginRegistrar.register
class MXNetTensor(Plugin):
    def __init__(self, load_mxnet_device=None, **kwargs):
        self.mxnet_device = load_mxnet_device

    def encode(self, obj, *args, **kwargs):
        if HAVE_MXNET and isinstance(obj, mxnet.nd.NDArray):
            with io.BytesIO() as memfile:
                np.save(memfile, obj.asnumpy())
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            return True, dict(__wrapyfi__=('mxnet.Tensor', obj_data))
        else:
            return False, None

    def decode(self, obj_type, obj_full, *args, **kwargs):
        if HAVE_MXNET and obj_type == 'mxnet.Tensor':
            with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
                return True, mxnet.nd.array(np.load(memfile), ctx=self.mxnet_device)
        else:
            return False, None

