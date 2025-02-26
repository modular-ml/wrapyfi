"""
Encoder and Decoder for TensorFlow Tensor Data via Wrapyfi.

This script provides mechanisms to encode and decode TensorFlow tensor data using Wrapyfi.
It utilizes base64 encoding to convert binary data into ASCII strings.

The script contains a class, `TensorflowTensor`, registered as a plugin to manage the
conversion of TensorFlow tensor data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - TensorFlow: An end-to-end open-source platform for machine learning (refer to https://www.tensorflow.org/install for installation instructions)
        Note: If TensorFlow is not available, HAVE_TENSORFLOW will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install tensorflow``  # Basic installation of TensorFlow
"""

import io
import base64

import numpy as np

from wrapyfi.utils.core_utils import *

try:
    import tensorflow

    HAVE_TENSORFLOW = True
except ImportError:
    HAVE_TENSORFLOW = False


@PluginRegistrar.register(
    types=None if not HAVE_TENSORFLOW else tensorflow.Tensor.__mro__[:-1]
)
class TensorflowTensor(Plugin):
    def __init__(self, **kwargs):
        """
        Initialize the TensorflowTensor plugin.
        """
        pass

    def encode(self, obj, *args, **kwargs):
        """
        Encode TensorFlow tensor data into a base64 ASCII string.

        :param obj: tensorflow.Tensor: The TensorFlow tensor data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name and encoded data string
        """
        with io.BytesIO() as memfile:
            np.save(memfile, obj.numpy())
            obj_data = base64.b64encode(memfile.getvalue()).decode("ascii")
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into TensorFlow tensor data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, tensorflow.Tensor]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - tensorflow.Tensor: The decoded TensorFlow tensor data
        """
        with io.BytesIO(base64.b64decode(obj_full[1].encode("ascii"))) as memfile:
            return True, tensorflow.convert_to_tensor(np.load(memfile))
