"""
Encoder and Decoder for Pint Quantity Data via Wrapyfi.

This script provides mechanisms to encode and decode Pint Quantity Data using Wrapyfi.
It utilizes base64 encoding and JSON serialization.

The script contains a class, `PintData`, registered as a plugin to manage the
conversion of Pint Quantity data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - Pint: A package to define, operate and manipulate physical quantities (refer to https://pint.readthedocs.io/en/stable/ for installation instructions)
        Note: If Pint is not available, HAVE_PINT will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install pint``  # Basic installation of Pint
"""

import base64
import json

from wrapyfi.utils import *

try:
    from pint import Quantity

    HAVE_PINT = True
except ImportError:
    HAVE_PINT = False


@PluginRegistrar.register(types=None if not HAVE_PINT else Quantity.__mro__[:-1])
class PintData(Plugin):
    def __init__(self, **kwargs):
        """
        Initialize the PintData plugin.
        """
        pass

    def encode(self, obj, *args, **kwargs):
        """
        Encode Pint Quantity data into a base64 ASCII string.

        :param obj: pint.Quantity: The Pint Quantity data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name, encoded data string, and object type
        """
        if isinstance(obj, Quantity):
            obj_type = "Quantity"
            obj_data = json.dumps(
                {"magnitude": obj.magnitude, "units": str(obj.units)}
            ).encode("ascii")
            obj_data = base64.b64encode(obj_data).decode("ascii")
            return True, dict(
                __wrapyfi__=(str(self.__class__.__name__), obj_data, obj_type)
            )
        else:
            # TypeError("Unknown object type: {}".format(obj_type))
            return False, {}

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into Pint Quantity data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string and object type
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, pint.Quantity]: A tuple containing:
            - bool: Indicating that the decoding was successful
            - pint.Quantity: The decoded Pint Quantity data
        """
        obj_data = base64.b64decode(obj_full[1].encode("ascii")).decode("ascii")
        obj_data = json.loads(obj_data)
        obj_type = obj_full[2]
        if obj_type == "Quantity":
            from pint import UnitRegistry

            ureg = UnitRegistry()
            obj = Quantity(
                obj_data["magnitude"], ureg.parse_expression(obj_data["units"])
            )
            return True, obj
        else:
            # TypeError("Unknown object type: {}".format(obj_type))
            return False, {}
