"""
Encoder and Decoder for PIL Image Data via Wrapyfi.

This script provides mechanisms to encode and decode PIL Image data using Wrapyfi.
It utilizes base64 encoding to convert binary data into ASCII strings, and also handles non-ASCII encodings for certain image formats.

The script contains a class, `PILImage`, registered as a plugin to manage the
conversion of PIL Image data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - PIL (Pillow): A Python Imaging Library that adds image processing capabilities to your Python interpreter (refer to https://pillow.readthedocs.io/en/stable/installation.html for installation instructions)
        Note: If PIL is not available, HAVE_PIL will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install Pillow``  # Basic installation of Pillow (PIL Fork)
"""

import io
import base64

from wrapyfi.utils.core_utils import *

try:
    from PIL import Image

    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


@PluginRegistrar.register(types=None if not HAVE_PIL else Image.Image.__mro__[:-1])
class PILImage(Plugin):
    def __init__(self, **kwargs):
        """
        Initialize the PILImage plugin.
        """
        pass

    def encode(self, obj, *args, **kwargs):
        """
        Encode PIL Image data into a base64 ASCII string.

        :param obj: Image.Image: The PIL Image data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name and encoded data string, with optional image size and mode for raw data
        """
        if obj.format is None:
            obj_data = base64.b64encode(obj.tobytes()).decode("ascii")
            return True, dict(
                __wrapyfi__=(str(self.__class__.__name__), obj_data, obj.size, obj.mode)
            )
        else:
            with io.BytesIO() as memfile:
                obj.save(memfile, format=obj.format)
                obj_data = memfile.getvalue().decode("latin1")
            return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into PIL Image data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string and optionally image size and mode for raw data
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, Image.Image]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - Image.Image: The decoded PIL Image data
        """
        if len(obj_full) == 4:
            with io.BytesIO(obj_full[1].encode("ascii")) as memfile:
                return True, Image.frombytes(
                    obj_full[3], obj_full[2], memfile.getvalue(), "raw"
                )
        else:
            with io.BytesIO(obj_full[1].encode("latin1")) as memfile:
                memfile.seek(0)
                return True, Image.open(memfile).copy()
