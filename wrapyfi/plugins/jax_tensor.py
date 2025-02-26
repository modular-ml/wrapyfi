"""
Encoder and Decoder for JAX Tensor Data via Wrapyfi.

This script provides mechanisms to encode and decode JAX tensor data using Wrapyfi.
It utilizes base64 encoding to convert binary data into ASCII strings.

The script contains a class, `JAXTensor`, registered as a plugin to manage the
conversion of JAX tensor data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - JAX: An open-source high-performance machine learning library (refer to https://github.com/google/jax for installation instructions)
        Note: If JAX is not available, HAVE_JAX will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install jax jaxlib``  # Basic installation of JAX
"""

import io
import base64

import numpy as np

from wrapyfi.utils.core_utils import *

try:
    import jax

    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False

try:
    # if jax 0.3.22 is installed, then jax.numpy is a module
    types = None if not HAVE_JAX else (jax.Array,)
except AttributeError:
    types = None if not HAVE_JAX else jax.numpy.DeviceArray.__mro__[:-1]


@PluginRegistrar.register(types=types)
class JAXTensor(Plugin):
    def __init__(self, **kwargs):
        """
        Initialize the JAXTensor plugin.
        """
        pass

    def encode(self, obj, *args, **kwargs):
        """
        Encode JAX tensor data into a base64 ASCII string.

        :param obj: jax.numpy.DeviceArray: The JAX tensor data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name and encoded data string
        """
        with io.BytesIO() as memfile:
            np.save(memfile, np.asarray(obj))
            obj_data = base64.b64encode(memfile.getvalue()).decode("ascii")
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into JAX tensor data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, jax.numpy.DeviceArray]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - jax.numpy.DeviceArray: The decoded JAX tensor data
        """
        with io.BytesIO(base64.b64decode(obj_full[1].encode("ascii"))) as memfile:
            return True, jax.numpy.array(np.load(memfile))
