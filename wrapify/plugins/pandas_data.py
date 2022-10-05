import io
import json
import base64

import numpy as np

from wrapify.utils import *

try:
    import pandas
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


@PluginRegistrar.register
class PandasData(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):
        if HAVE_PANDAS and isinstance(obj, pandas.DataFrame) or isinstance(obj, pandas.Series):
            with io.BytesIO() as memfile:
                obj.to_json(memfile, orient="records")
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            return True, dict(__wrapify__=('pandas.Data', obj_data))
        else:
            return False, None

    def decode(self, obj_type, obj_full, *args, **kwargs):
        if HAVE_PANDAS and obj_type == 'pandas.Data':
            with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
                return True, pandas.read_json(memfile, orient="records")
        else:
            return False, None

