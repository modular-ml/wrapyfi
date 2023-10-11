"""
Encoder and Decoder for MXNet Tensor Data via Wrapyfi.

This script provides mechanisms to encode and decode MXNet tensor data using Wrapyfi.
It leverages base64 encoding to convert binary data into ASCII strings.

The script contains a class, `MXNetTensor`, registered as a plugin to manage the
conversion of MXNet tensor data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - MXNet: A deep learning framework designed for both efficiency and flexibility (refer to https://mxnet.apache.org/get_started for installation instructions)
        Note: If MXNet is not available, HAVE_MXNET will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install mxnet``  # Basic installation of MXNet
"""

import io
import base64
from functools import lru_cache

import numpy as np

from wrapyfi.utils import *


try:
    import mxnet
    HAVE_MXNET = True
except ImportError:
    HAVE_MXNET = False


def mxnet_device_to_str(device):
    """
    Convert an MXNet device to a string representation.

    :param device: Union[str, mxnet.Context, dict]: Various possible types representing an MXNet device as mxnet.Context or str. Also accepts a dictionary mapping encoded device strings to decoding devices
    :return: Union[str, dict]: A string or dictionary representing the MXNet device
    """
    if device is None:
        return 'cpu:0'
    elif isinstance(device, dict):
        device_rets = {}
        for k, v in device.items():
            device_rets[mxnet_device_to_str(k)] = mxnet_device_to_str(v)
        return device_rets
    elif isinstance(device, mxnet.Context):
        return f'{device.device_type.replace("gpu", "cuda")}:{device.device_id}'
    elif isinstance(device, str):
        return device.replace("gpu", "cuda")
    else:
        raise ValueError(f'Unknown device type {type(device)}')


@lru_cache(maxsize=None)
def mxnet_str_to_device(device):
    """
    Convert a string to an MXNet device.

    :param device: str: A string representing an MXNet device
    :return: mxnet.Context: An MXNet context representing the device
    """
    if device is None:
        return mxnet.cpu()
    elif isinstance(device, mxnet.Context):
        return device
    elif isinstance(device, str):
        try:
            device_type, device_id = device.split(':')
        except ValueError:
            device_type = device
            device_id = 0
        return mxnet.Context(device_type.replace("cuda", "gpu"), int(device_id))
    else:
        raise ValueError(f'Unknown device type {type(device)}')


@PluginRegistrar.register(types=None if not HAVE_MXNET else mxnet.nd.NDArray.__mro__[:-1])
class MXNetTensor(Plugin):
    def __init__(self, load_mxnet_device=None, map_mxnet_devices=None, **kwargs):
        """
        Initialize the MXNetTensor plugin.

        :param load_mxnet_device: Union[mxnet.Context, str]: Default MXNet device to load tensors onto
        :param map_mxnet_devices: dict: [Optional] A dictionary mapping encoded device strings to decoding devices
        """
        self.map_mxnet_devices = map_mxnet_devices or {}
        if load_mxnet_device is not None:
            self.map_mxnet_devices['default'] = load_mxnet_device
        self.map_mxnet_devices = mxnet_device_to_str(self.map_mxnet_devices)

    def encode(self, obj, *args, **kwargs):
        """
        Encode MXNet tensor data into a base64 ASCII string.

        :param obj: mxnet.nd.NDArray: The MXNet tensor data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name, encoded data string, and device string.
        """
        with io.BytesIO() as memfile:
            np.save(memfile, obj.asnumpy())
            obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
        obj_device = mxnet_device_to_str(obj.context)
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_device))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into MXNet tensor data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string and device string
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, mxnet.nd.NDArray]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - mxnet.nd.NDArray: The decoded MXNet tensor data
        """
        with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
            obj_device = self.map_mxnet_devices.get(obj_full[2], self.map_mxnet_devices.get('default', None))
            if obj_device is not None:
                obj_device = mxnet_str_to_device(obj_device)
                return True, mxnet.nd.array(np.load(memfile), ctx=obj_device)
            else:
                return True, mxnet.nd.array(np.load(memfile))
