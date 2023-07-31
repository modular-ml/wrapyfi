import pickle
import base64

from wrapyfi.utils import *

try:
    import pyarrow as pa
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False


@PluginRegistrar.register(types=None if not HAVE_PYARROW else pa.StructArray.__mro__[:-1])
class PyArrowArray(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):
        buffers = []
        obj_data = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
        obj_buffers = list(map(lambda x: base64.b64encode(memoryview(x)).decode('ascii'), buffers))
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data.decode('latin1'), *obj_buffers))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        obj_data = obj_full[1].encode('latin1')
        obj_buffers = list(map(lambda x: base64.b64decode(x.encode('ascii')), obj_full[2:]))
        obj_data = bytearray(obj_data)
        for buf in obj_buffers:
            obj_data += buf
        return True, pickle.loads(obj_data, buffers=obj_buffers)
