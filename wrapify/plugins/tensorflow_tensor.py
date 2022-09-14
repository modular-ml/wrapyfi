import io
import json
import base64

import numpy as np

from wrapify.utils import *

try:
    import tensorflow
    HAVE_TENSORFLOW = True
except ImportError:
    HAVE_TENSORFLOW = False


@PluginRegistrar.register
class TensorflowTensor(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):
        if HAVE_TENSORFLOW and isinstance(obj, tensorflow.Tensor):
            with io.BytesIO() as memfile:
                np.save(memfile, obj.numpy())
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            return True, dict(__wrapify__=('tensorflow.Tensor', obj_data))
        else:
            return False, None

    def decode(self, obj_type, obj_full, *args, **kwargs):
        if HAVE_TENSORFLOW and obj_type == 'tensorflow.Tensor':
            with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
                return True, tensorflow.convert_to_tensor(np.load(memfile))
        else:
            return False, None

