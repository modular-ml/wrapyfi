import io
import json
import base64

import numpy as np

from wrapyfi.utils import *

try:
    import jax
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False


@PluginRegistrar.register(types=None if not HAVE_JAX else jax.numpy.DeviceArray.__mro__[:-1])
class JAXTensor(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):
        with io.BytesIO() as memfile:
            np.save(memfile, np.asarray(obj))
            obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
            return True, jax.numpy.array(np.load(memfile))

