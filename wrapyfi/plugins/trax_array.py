"""
Encoder and Decoder for Trax Array Data via Wrapyfi.

This script provides mechanisms to encode and decode Trax Array Data using Wrapyfi.
It utilizes base64 encoding and pickle serialization.

The script contains a class, `TraxArray`, registered as a plugin to manage the
conversion of Trax Array data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - Trax: A deep learning library that focuses on clear code and speed (refer to https://trax-ml.readthedocs.io/en/latest/notebooks/trax_intro.html for installation instructions)
        Note: If Trax is not available, HAVE_TRAX will be set to False and
        the plugin will be registered with no types. Trax uses JAX or TensorFlow-NumPy as its backend,
        so they must be installed as well. Trax installs JAX as a dependency, but TensorFlow must be installed separately.

    You can install the necessary packages using pip:
        ``pip install trax``  # Basic installation of Trax
"""

import pickle
import base64

from wrapyfi.utils import *

try:
    import trax
    import jax
    import jaxlib.xla_extension

    HAVE_TRAX = True
except ImportError:
    HAVE_TRAX = False


@PluginRegistrar.register(
    types=None if not HAVE_TRAX else jaxlib.xla_extension.ArrayImpl.__mro__[:-1]
)
class TraxArray(Plugin):
    def __init__(self, **kwargs):
        """
        Initialize the TraxArray plugin.
        """
        pass

    def encode(self, obj, *args, **kwargs):
        """
        Encode Trax Array data using pickle and base64.

        :param obj: jaxlib.xla_extension.ArrayImpl: The Trax Array data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name, pickled data string, and any buffer data
        """
        buffers = []
        obj_data = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
        obj_buffers = list(
            map(lambda x: base64.b64encode(memoryview(x)).decode("ascii"), buffers)
        )
        return True, dict(
            __wrapyfi__=(
                str(self.__class__.__name__),
                obj_data.decode("latin1"),
                *obj_buffers,
            )
        )

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a pickled and base64 encoded string back into Trax Array data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the pickled data string and any buffer data
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, pa.StructArray]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - jaxlib.xla_extension.ArrayImpl: The decoded Trax Array data
        """
        obj_data = obj_full[1].encode("latin1")
        obj_buffers = list(
            map(lambda x: base64.b64decode(x.encode("ascii")), obj_full[2:])
        )
        obj_data = bytearray(obj_data)
        for buf in obj_buffers:
            obj_data += buf
        return True, pickle.loads(obj_data, buffers=obj_buffers)
