import io
import json
import base64

import numpy as np

from wrapyfi.utils import *

try:
    import pandas
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


@PluginRegistrar.register(types=None if not HAVE_PANDAS else pandas.DataFrame.__mro__[:-1] + pandas.Series.__mro__[:-1])
class PandasData(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):
        with io.BytesIO() as memfile:
            if isinstance(obj, pandas.DataFrame):
                obj_type = 'DataFrame'
                obj.to_json(memfile, orient="split")
            elif isinstance(obj, pandas.Series):
                obj_type = 'Series'
                obj.to_frame().to_json(memfile, orient="split")
            obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_type))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
            obj = pandas.read_json(memfile, orient="split")
            obj_type = obj_full[2]
            if obj_type == 'Series':
                obj = obj.iloc[:, 0]
        return True, obj

