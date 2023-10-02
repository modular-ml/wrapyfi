# astropy_data.py
import io
import base64

from wrapyfi.utils import *

try:
    from astropy.table import Table
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False


@PluginRegistrar.register(types=None if not HAVE_ASTROPY else Table.__mro__[:-1])
class AstropyData(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):
        memfile = io.BytesIO()
        obj.write(memfile, format='fits')  # using a binary format
        memfile.seek(0)
        obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
        memfile.close()
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        encoded_str = obj_full[1]
        if isinstance(encoded_str, str):  # Ensure the string is encoded to bytes before decoding it
            encoded_str = encoded_str.encode('ascii')
        with io.BytesIO(base64.b64decode(encoded_str)) as memfile:
            memfile.seek(0)  # Ensure cursor is at the beginning of the BytesIO object
            obj = Table.read(memfile, format='fits')  # Corrected to 'fits' to match the encoding format
        return True, obj