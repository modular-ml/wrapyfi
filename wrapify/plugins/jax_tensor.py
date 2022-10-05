import io
import json
import base64

import numpy as np

from wrapify.utils import *

try:
    import jax
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False


@PluginRegistrar.register
class JAXTensor(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):
        if HAVE_JAX and isinstance(obj, jax.numpy.DeviceArray):
            with io.BytesIO() as memfile:
                np.save(memfile, np.asarray(obj))
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            return True, dict(__wrapify__=('jax.Tensor', obj_data))
        else:
            return False, None

    def decode(self, obj_type, obj_full, *args, **kwargs):
        if HAVE_JAX and obj_type == 'jax.Tensor':
            with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
                return True, jax.numpy.array(np.load(memfile))
        else:
            return False, None

