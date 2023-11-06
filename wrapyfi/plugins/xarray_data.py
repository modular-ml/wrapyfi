"""
Encoder and Decoder for XArray DataArray/Dataset Data via Wrapyfi.

This script provides mechanisms to encode and decode XArray Data using Wrapyfi.
It utilizes base64 encoding and JSON serialization.

The script contains a class, `XArrayData`, registered as a plugin to manage the
conversion of XArray Data (DataArray and Dataset) (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - pandas: A data structures library for data analysis, time series, and statistics (refer to https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html for installation instructions)
    - XArray: N-D labeled arrays and datasets in Python (refer to http://xarray.pydata.org/en/stable/getting-started-guide/installing.html for installation instructions)
        Note: If XArray is not available, HAVE_XARRAY will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install "pandas<2.0" xarray``  # Basic installation of XArray
"""

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
        """
        Initialize the XArrayData plugin.
        """
        pass

    def encode(self, obj, *args, **kwargs):
        """
        Encode XArray Data using JSON and base64.

        :param obj: Union[xr.DataArray, xr.Dataset]: The XArray Data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name, encoded data string, data type, and object name
        """
        obj_type = 'Dataset'
        obj_name = None
        if isinstance(obj, xr.DataArray):
            obj_dict = obj.to_dataset().to_dict()
            obj_type = 'DataArray'
            obj_name = obj.name
        elif isinstance(obj, xr.Dataset):
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
        """
        Decode a JSON and base64 encoded string back into XArray Data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string, data type, and object name
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, Union[xr.DataArray, xr.Dataset]]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - Union[xr.DataArray, xr.Dataset]: The decoded XArray Data
        """
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