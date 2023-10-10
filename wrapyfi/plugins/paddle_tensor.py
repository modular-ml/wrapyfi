"""
Encoder and Decoder for PaddlePaddle Tensor Data via Wrapyfi.

This script provides mechanisms to encode and decode PaddlePaddle tensor data using Wrapyfi.
It utilizes base64 encoding to convert binary data into ASCII strings.

The script contains a class, `PaddleTensor`, registered as a plugin to manage the
conversion of PaddlePaddle tensor data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - PaddlePaddle: An open-source deep learning platform developed by Baidu (refer to https://www.paddlepaddle.org.cn/en/install/quick for installation instructions)
        Note: If PaddlePaddle is not available, HAVE_PADDLE will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install paddlepaddle``  # Basic installation of PaddlePaddle
"""

import io
import base64
from functools import lru_cache

import numpy as np

from wrapyfi.utils import *


try:
    import paddle
    from paddle import Tensor
    HAVE_PADDLE = True
except ImportError:
    HAVE_PADDLE = False


def paddle_device_to_str(device):
    """
    Convert a PaddlePaddle device to a string representation.

    :param device: Union[str, paddle.fluid.libpaddle.Place, dict]: Various possible types representing a PaddlePaddle device
    :return: Union[str, dict]: A string or dictionary representing the PaddlePaddle device
    """
    if device is None:
        return paddle.get_device()
    elif isinstance(device, dict):
        device_rets = {}
        for k, v in device.items():
            device_rets[paddle_device_to_str(k)] = paddle_device_to_str(v)
        return device_rets
    elif isinstance(device, paddle.fluid.libpaddle.Place):
        return str(device)[6:-1].replace("cuda", "gpu")
    elif isinstance(device, str):
        return device.replace("cuda", "gpu")
    else:
        raise ValueError(f'Unknown device type {type(device)}')


@lru_cache(maxsize=None)
def paddle_str_to_device(device):
    """
    Convert a string to a PaddlePaddle device.

    :param device: str: A string representing a PaddlePaddle device
    :return: paddle.fluid.libpaddle.Place: A PaddlePaddle place representing the device
    """
    if device is None:
        return paddle.get_device()
    elif isinstance(device, paddle.fluid.libpaddle.Place):
        return device
    elif isinstance(device, str):
        try:
            device_type, device_id = device.split(':')
        except ValueError:
            device_type = device
        return paddle.device._convert_to_place(device_type.replace("cuda", "gpu"))
    else:
        raise ValueError(f'Unknown device type {type(device)}')


@PluginRegistrar.register(types=None if not HAVE_PADDLE else paddle.Tensor.__mro__[:-1])
class PaddleTensor(Plugin):
    def __init__(self, load_paddle_device=None, map_paddle_devices=None, **kwargs):
        """
        Initialize the PaddleTensor plugin.

        :param load_paddle_device: Union[paddle.fluid.libpaddle.Place, str]: Default PaddlePaddle device to load tensors onto
        :param map_paddle_devices: dict: A dictionary mapping encoded device strings to decoding devices
        """
        self.map_paddle_devices = map_paddle_devices or {}
        if load_paddle_device is not None:
            self.map_paddle_devices['default'] = load_paddle_device
        self.map_paddle_devices = paddle_device_to_str(self.map_paddle_devices)

    def encode(self, obj, *args, **kwargs):
        """
        Encode PaddlePaddle tensor data into a base64 ASCII string.

        :param obj: paddle.Tensor: The PaddlePaddle tensor data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name, encoded data string, and device string
        """
        with io.BytesIO() as memfile:
            paddle.save(obj, memfile)
            obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
        obj_device = paddle_device_to_str(obj.place)
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_device))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into PaddlePaddle tensor data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string and device string
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, paddle.Tensor]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - paddle.Tensor: The decoded PaddlePaddle tensor data
        """
        with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
            obj_device = self.map_paddle_devices.get(obj_full[2], self.map_paddle_devices.get('default', None))
            if obj_device is not None:
                obj_device = paddle_str_to_device(obj_device)
                return True, paddle.Tensor(paddle.load(memfile), place=obj_device)
            else:
                return True, paddle.load(memfile)

