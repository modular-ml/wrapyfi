"""
Encoder and Decoder for PyTorch Tensor Data via Wrapyfi.

This script provides mechanisms to encode and decode PyTorch tensor data using Wrapyfi.
It utilizes base64 encoding to convert binary data into ASCII strings.

The script contains a class, `PytorchTensor`, registered as a plugin to manage the
conversion of PyTorch tensor data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - PyTorch: An open-source machine learning library developed by Facebook's AI Research lab (refer to https://pytorch.org/get-started/locally/ for installation instructions)
        Note: If PyTorch is not available, HAVE_TORCH will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install torch``  # Basic installation of PyTorch
"""

import io
import base64

from wrapyfi.utils import *

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@PluginRegistrar.register(types=None if not HAVE_TORCH else torch.Tensor.__mro__[:-1])
class PytorchTensor(Plugin):
    def __init__(self, load_torch_device=None, map_torch_devices=None, **kwargs):
        """
        Initialize the PytorchTensor plugin.

        :param load_torch_device: str: Default PyTorch device to load tensors onto
        :param map_torch_devices: dict: A dictionary mapping encoded device strings to decoding devices
        """
        self.map_torch_devices = map_torch_devices or {}
        if load_torch_device is not None:
            self.map_torch_devices['default'] = load_torch_device

    def encode(self, obj, *args, **kwargs):
        """
        Encode PyTorch tensor data into a base64 ASCII string.

        :param obj: torch.Tensor: The PyTorch tensor data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name, encoded data string, and device string
        """
        with io.BytesIO() as memfile:
            torch.save(obj, memfile)
            obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
        obj_device = str(obj.device)
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_device))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into PyTorch tensor data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string and device string
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, torch.Tensor]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - torch.Tensor: The decoded PyTorch tensor data
        """
        with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
            obj_device = self.map_torch_devices.get(obj_full[2], self.map_torch_devices.get('default', None))
            if obj_device is not None:
                return True, torch.load(memfile, map_location=obj_device)
            else:
                return True, torch.load(memfile)

