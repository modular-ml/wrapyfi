"""
Encoder and Decoder for pandas Series/Dataframes Data via Wrapyfi.

This script provides mechanisms to encode and decode pandas data using Wrapyfi.
It utilizes base64 encoding to convert binary data into ASCII strings.

The script contains a class, `PandasData`, registered as a plugin to manage the
conversion of pandas data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - Pandas: A data structures library for data analysis, time series, and statistics (refer to https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html for installation instructions)
        Note: If pandas is not available, HAVE_PANDAS will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install "pandas<2.0"``  # Basic installation of Pandas
"""

import io
import base64

import numpy as np

from wrapyfi.utils import *

try:
    import pandas
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


@PluginRegistrar.register(types=None if not HAVE_PANDAS else pandas.DataFrame.__mro__[:-1] + pandas.Series.__mro__[:-1])
class PandasData(Plugin):
    def __init__(self, **kwargs):
        """
        Initialize the PandasData plugin.
        """
        pass

    def encode(self, obj, *args, **kwargs):
        """
        Encode pandas data into a base64 ASCII string.

        :param obj: Union[pandas.DataFrame, pandas.Series]: The pandas data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name, encoded data string, and object type
        """
        with io.BytesIO() as memfile:
            if isinstance(obj, pandas.DataFrame):
                obj_type = 'DataFrame'
                obj.to_json(memfile, orient="split")
            elif isinstance(obj, pandas.Series):
                obj_type = 'Series'
                obj.to_frame().to_json(memfile, orient="split")
            obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_type))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into pandas data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string and object type
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, Union[pandas.DataFrame, pandas.Series]]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - Union[pandas.DataFrame, pandas.Series]: The decoded pandas data
        """
        with io.BytesIO(base64.b64decode(obj_full[1].encode('ascii'))) as memfile:
            obj = pandas.read_json(memfile, orient="split")
            obj_type = obj_full[2]
            if obj_type == 'Series':
                obj = obj.iloc[:, 0]
        return True, obj

