"""
Encoder and Decoder for CuPy Array Data via Wrapyfi.

This script provides mechanisms to encode and decode CuPy array data using Wrapyfi.
It leverages base64 encoding to convert binary data into ASCII strings.

The script contains a class, CuPyArray, registered as a plugin to manage the
conversion of CuPy array data (if available) between its original and encoded forms. CuPy only supports GPU devices,
specifically NVIDIA CUDA devices. Installing CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit) is required
to use CuPy.

Requirements:
- Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
- CuPy: A GPU-accelerated library for numerical computations with a NumPy-compatible interface (refer to https://docs.cupy.dev/en/stable/install.html for installation instructions)
Note: If CuPy is not available, HAVE_CUPY will be set to False and the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install cupy-cuda12x``  # Basic installation of CuPy. Replace 12x with your CUDA version e.g., cupy-cuda11x
"""

import io
import base64

import numpy as np

from wrapyfi.utils.core_utils import *

try:
    import cupy

    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False


def cupy_device_to_str(device):
    """
    Convert a CuPy device to a string representation.

    :param device: Union[cupy.cuda.Device, int]: The CuPy device
    :return: str: A string representing the CuPy device
    """
    if isinstance(device, (cupy.cuda.Device, int)):
        return f"cuda:{int(device)}"
    elif isinstance(device, str):
        return device
    else:
        raise ValueError(f"Unknown device type {type(device)}")


def cupy_str_to_device(device_str):
    """
    Convert a string to a CuPy device.

    :param device_str: str: A string representing a CuPy device
    :return: cupy.cuda.Device: A CuPy device
    """
    if isinstance(device_str, str) and device_str.startswith("cuda:"):
        device_id = int(device_str.split(":")[1])
        return cupy.cuda.Device(device_id)
    elif isinstance(device_str, str) and device_str.startswith("cpu:"):
        raise ValueError("CuPY does not support CPU devices")
    else:
        raise ValueError(f"Invalid device string {device_str}")


@PluginRegistrar.register(types=None if not HAVE_CUPY else cupy.ndarray.__mro__[:-1])
class CuPyArray(Plugin):
    def __init__(self, load_cupy_device=None, map_cupy_devices=None, **kwargs):
        """
        Initialize the CuPy plugin.

        :param load_cupy_device: cupy.cuda.Device or str: Default CuPy device to load tensors onto
        :param map_cupy_devices: dict: [Optional] A dictionary mapping encoded device strings to decoding devices
        """
        self.map_cupy_devices = map_cupy_devices or {}
        if load_cupy_device is not None:
            self.map_cupy_devices["default"] = load_cupy_device
        self.map_cupy_devices = {
            k: cupy_device_to_str(v) for k, v in self.map_cupy_devices.items()
        }

    def encode(self, obj, *args, **kwargs):
        """
        Encode CuPy array into a base64 ASCII string.

        :param obj: cupy.ndarray: The CuPy array to encode
        :return: Tuple[bool, dict]
        """
        with io.BytesIO() as memfile:
            np.save(memfile, cupy.asnumpy(obj))
            obj_data = base64.b64encode(memfile.getvalue()).decode("ascii")
        obj_device = cupy_device_to_str(obj.device)
        return True, dict(
            __wrapyfi__=(str(self.__class__.__name__), obj_data, obj_device)
        )

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into CuPy array.

        :param obj_full: tuple: A tuple containing the encoded data string and device string
        :return: Tuple[bool, cupy.ndarray]
        """
        with io.BytesIO(base64.b64decode(obj_full[1].encode("ascii"))) as memfile:
            obj_device_str = self.map_cupy_devices.get(
                obj_full[2], self.map_cupy_devices.get("default", "cuda:0")
            )
            obj_device = cupy_str_to_device(obj_device_str)
            with obj_device:
                return True, cupy.array(np.load(memfile))
