"""
Encoder and Decoder for PyArrow StructArray Data via Wrapyfi.

This script provides mechanisms to encode and decode PyArrow StructArray Data using Wrapyfi.
It utilizes base64 encoding and pickle serialization.

The script contains a class, `PyArrowArray`, registered as a plugin to manage the
conversion of PyArrow StructArray data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - PyArrow: A cross-language development platform for in-memory data (refer to https://arrow.apache.org/install/ for installation instructions)
        Note: If PyArrow is not available, HAVE_PYARROW will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install pyarrow``  # Basic installation of PyArrow
"""

import pickle
import base64

from wrapyfi.utils import *

try:
    import pyarrow as pa

    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False


@PluginRegistrar.register(
    types=None if not HAVE_PYARROW else pa.StructArray.__mro__[:-1]
)
class PyArrowArray(Plugin):
    def __init__(self, **kwargs):
        """
        Initialize the PyArrowArray plugin.
        """
        pass

    def encode(self, obj, *args, **kwargs):
        """
        Encode PyArrow StructArray data using pickle and base64.

        :param obj: pa.StructArray: The PyArrow StructArray data to encode
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
        Decode a pickled and base64 encoded string back into PyArrow StructArray data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the pickled data string and any buffer data
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, pa.StructArray]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - pa.StructArray: The decoded PyArrow StructArray data
        """
        obj_data = obj_full[1].encode("latin1")
        obj_buffers = list(
            map(lambda x: base64.b64decode(x.encode("ascii")), obj_full[2:])
        )
        obj_data = bytearray(obj_data)
        for buf in obj_buffers:
            obj_data += buf
        return True, pickle.loads(obj_data, buffers=obj_buffers)
