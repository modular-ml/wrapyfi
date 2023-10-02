import json
import base64
from datetime import datetime

import numpy as np

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
        obj_type = 'Dataset'
        obj_name = None
        if isinstance(obj, xr.DataArray):
            obj_dict = obj.to_dataset().to_dict()
            obj_type = 'DataArray'
            obj_name = obj.name
        else:
            obj_dict = obj.to_dict()

        def traverse_and_convert(obj):
            if isinstance(obj, list):
                return [traverse_and_convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: traverse_and_convert(value) for key, value in obj.items()}
            elif isinstance(obj, np.datetime64):
                return {'data': str(obj), 'dtype': str(obj.dtype)}
            elif isinstance(obj, datetime):
                return {'data': obj.isoformat(), 'dtype': 'datetime'}
            else:
                return obj

        converted_obj_dict = traverse_and_convert(obj_dict)
        obj_json = json.dumps(converted_obj_dict)
        obj_data = base64.b64encode(obj_json.encode('ascii')).decode('ascii')

        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_type, obj_name))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        obj_data = base64.b64decode(obj_full[1].encode('ascii')).decode('ascii')
        xarray_type = obj_full[2]
        xarray_name = obj_full[3]

        def traverse_and_reconvert(obj):
            if isinstance(obj, list):
                return [traverse_and_reconvert(item) for item in obj]
            elif isinstance(obj, dict):
                if 'data' in obj and 'dtype' in obj:
                    if obj['dtype'] == 'datetime':
                        return datetime.fromisoformat(obj['data'])
                    else:
                        return np.datetime64(obj['data'], obj['dtype'])
                return {key: traverse_and_reconvert(value) for key, value in obj.items()}
            else:
                return obj

        obj_dict = json.loads(obj_data)
        reconverted_obj_dict = traverse_and_reconvert(obj_dict)

        if xarray_type == 'DataArray':
            dataset = xr.Dataset.from_dict(reconverted_obj_dict)
            data_array = dataset[xarray_name]
            return True, data_array
        else:
            return True, xr.Dataset.from_dict(reconverted_obj_dict)