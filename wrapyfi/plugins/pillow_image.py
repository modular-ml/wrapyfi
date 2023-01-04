import io
import json
import base64

from wrapyfi.utils import *

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


@PluginRegistrar.register(types=None if not HAVE_PIL else Image.Image.__mro__[:-1])
class PILImage(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):

        if obj.format is None:
            obj_data = base64.b64encode(obj.tobytes()).decode('ascii')
            return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj.size, obj.mode))
        else:
            with io.BytesIO() as memfile:
                obj.save(memfile, format=obj.format)
                obj_data = memfile.getvalue().decode('latin1')
            return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        if len(obj_full) == 4:
            with io.BytesIO(obj_full[1].encode('ascii')) as memfile:
                return True, Image.frombytes(obj_full[3], obj_full[2], memfile.getvalue(), "raw")
        else:
            with io.BytesIO(obj_full[1].encode('latin1')) as memfile:
                memfile.seek(0)
                return True, Image.open(memfile).copy()

