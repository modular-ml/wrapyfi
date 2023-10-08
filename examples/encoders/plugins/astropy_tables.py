"""
Encoder and Decoder for Astropy Table Data via Wrapyfi.

This script provides a mechanism to encode and decode Astropy table data using
Wrapyfi. It makes use of base64 encoding to transform binary data into ASCII strings,
which are more easily transmitted or stored.

The script contains a class, `AstropyData`, registered as a plugin to handle the
conversion of Astropy data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (Refer to the Wrapyfi documentation for installation instructions)
    - Astropy: A library for Astronomy computations and data handling in Python.
        Note: If Astropy is not available, HAVE_ASTROPY will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install wrapyfi astropy``
"""

import io
import base64

from wrapyfi.utils import *

try:
    from astropy.table import Table
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False


class AstropyData(Plugin):
    """
    A Wrapyfi plugin for encoding and decoding Astropy table data.

    This class provides methods to convert Astropy table data to an encoded form
    and decode it back to its original form. The data is encoded using base64
    to transform binary data into ASCII strings suitable for transmission or storage.
    """

    def __init__(self, **kwargs):
        """
        Initialize the AstropyData plugin.
        """

    def encode(self, obj, *args, **kwargs):
        """
        Encode Astropy table data into a base64 ASCII string.

        :param obj: Table: The Astropy table data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)

        :return: tuple: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name and the encoded data string
        """
        memfile = io.BytesIO()
        obj.write(memfile, format='fits')
        memfile.seek(0)
        obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
        memfile.close()
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into Astropy table data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)

        :return: Tuple[bool, astropy.Table]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - Table: The decoded Astropy table data
        """
        encoded_str = obj_full[1]
        if isinstance(encoded_str, str):
            encoded_str = encoded_str.encode('ascii')
        with io.BytesIO(base64.b64decode(encoded_str)) as memfile:
            memfile.seek(0)
            obj = Table.read(memfile, format='fits')
        return True, obj
