import base64
import json

import numpy as np

from wrapyfi.encoders import JsonEncoder, JsonDecodeHook
from wrapyfi.utils import *

try:
    import xarray as xr

    HAVE_XARRAY = True
except ImportError:
    HAVE_XARRAY = False


@PluginRegistrar.register(types=None if not HAVE_XARRAY else xr.DataArray.__mro__[:-1] + xr.Dataset.__mro__[:-1])
class XArrayData(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):
        if isinstance(obj, xr.DataArray):
            obj_dict = obj.to_dataset().to_dict()
        else:  # obj is an instance of xr.Dataset
            obj_dict = obj.to_dict()

        # convert datetime64 objects to strings
        for var_name, var_data in obj_dict.get('coords', {}).items():
            data_list = var_data.get('data', [])
            if data_list and isinstance(data_list[0], np.datetime64):
                var_data['data'] = [str(item) for item in data_list]

        obj_json = json.dumps(obj_dict, cls=JsonEncoder)
        obj_data = base64.b64encode(obj_json.encode('ascii')).decode('ascii')

        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        obj_data = base64.b64decode(obj_full[1].encode('ascii')).decode('ascii')
        obj_dict = json.loads(obj_data, object_hook=JsonDecodeHook().object_hook)

        # Convert string back to datetime64
        for var_name, var_data in obj_dict.get('coords', {}).items():
            data_list = var_data.get('data', [])
            if data_list and 'T' in data_list[0]:  # Rough check to see if it's a datetime string
                var_data['data'] = [np.datetime64(item) for item in data_list]

        return True, xr.Dataset.from_dict(obj_dict)