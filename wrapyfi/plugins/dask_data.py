"""
Encoder and Decoder for Dask Arrays/Dataframes Data via Wrapyfi.

This script provides mechanisms to encode and decode Dask data using Wrapyfi.
It utilizes base64 encoding to convert binary data into ASCII strings.

The script contains a class, `DaskData`, registered as a plugin to manage the
conversion of Dask data (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - Dask: A flexible library for parallel computing in Python (refer to https://docs.dask.org/en/latest/install.html for installation instructions)
    - pandas: A data structures library for data analysis, time series, and statistics (refer to https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html for installation instructions)
        Note: If Dask or pandas is not available, HAVE_DASK will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install "pandas<2.0" dask[complete]``  # Basic installation of Dask and pandas
"""

import io
import base64

import numpy as np

from wrapyfi.utils import *

try:
    import dask.dataframe as dd
    import dask.array as da
    import pandas as pd

    HAVE_DASK = True
except ImportError:
    HAVE_DASK = False


@PluginRegistrar.register(
    types=(
        None
        if not HAVE_DASK
        else dd.DataFrame.__mro__[:-1] + dd.Series.__mro__[:-1] + da.Array.__mro__[:-1]
    )
)
class DaskData(Plugin):
    def __init__(self, **kwargs):
        """
        Initialize the DaskData plugin.
        """
        pass

    def encode(self, obj, *args, **kwargs):
        """
        Encode Dask data into a base64 ASCII string.

        :param obj: Union[dd.DataFrame, dd.Series, da.Array]: The Dask data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name, encoded data string, and object type
        """
        obj_type = None
        obj_partitions = 1
        with io.BytesIO() as memfile:
            if isinstance(obj, dd.DataFrame):
                pandas_df = obj.compute().reset_index()
                memfile.write(pandas_df.to_json(orient="records").encode("ascii"))
                obj_partitions = obj.npartitions
                obj_type = "DataFrame"
            elif isinstance(obj, dd.Series):
                pandas_ds = obj.compute().reset_index()
                memfile.write(pandas_ds.to_json(orient="records").encode("ascii"))
                obj_partitions = obj.npartitions
                obj_type = "Series"
            elif isinstance(obj, da.Array):
                np.save(memfile, obj.compute(), allow_pickle=True)
                obj_type = "Array"
            memfile.seek(0)
            obj_data = base64.b64encode(memfile.read()).decode("ascii")
        return True, dict(
            __wrapyfi__=(
                str(self.__class__.__name__),
                obj_data,
                obj_type,
                obj_partitions,
            )
        )

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a base64 ASCII string back into Dask data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string and object type
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, Union[dd.DataFrame, dd.Series, da.Array]]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - Union[dd.DataFrame, da.Array]: The decoded Dask data
        """
        obj_data = base64.b64decode(obj_full[1].encode("ascii"))
        obj_type = obj_full[2]
        obj_partitions = obj_full[3]
        with io.BytesIO(obj_data) as memfile:
            if obj_type == "DataFrame":
                pandas_df = pd.read_json(
                    memfile.read().decode("ascii"), orient="records"
                )
                pandas_df.set_index(
                    "index", inplace=True
                )  # Set the index back after reading from JSON
                return True, dd.from_pandas(pandas_df, npartitions=obj_partitions)
            elif obj_type == "Series":
                pandas_ds = pd.read_json(
                    memfile.read().decode("ascii"), orient="records"
                )
                pandas_ds.set_index(
                    "index", inplace=True
                )  # Set the index back after reading from JSON
                return True, dd.from_pandas(pandas_ds, npartitions=obj_partitions)
            elif obj_type == "Array":
                np_array = np.load(memfile, allow_pickle=True)
                return True, da.from_array(np_array, chunks=np_array.shape)
