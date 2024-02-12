"""
Encoder and Decoder for Zarr Array/Group Data via Wrapyfi.

This script provides mechanisms to encode and decode Zarr Data using Wrapyfi.
It utilizes base64 encoding and zip compression.

The script contains a class, `ZarrData`, registered as a plugin to manage the
conversion of Zarr Data (Array and Group) (if available) between its original and encoded forms.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - Zarr: A format for the storage of chunked, compressed, N-dimensional arrays (refer to https://zarr.readthedocs.io/en/stable/ for installation instructions)
        Note: If Zarr is not available, HAVE_ZARR will be set to False and
        the plugin will be registered with no types.

    You can install the necessary packages using pip:
        ``pip install zarr``  # Basic installation of Zarr
"""

import tempfile
import io
import base64

from wrapyfi.utils import *

try:
    import zipfile
    import zarr

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@PluginRegistrar.register(
    types=None if not HAVE_ZARR else zarr.Array.__mro__[:-1] + zarr.Group.__mro__[:-1]
)
class ZarrData(Plugin):
    def __init__(self, **kwargs):
        """
        Initialize the ZarrData plugin.
        """
        pass

    def encode(self, obj, *args, **kwargs):
        """
        Encode Zarr Data using zip compression and base64.

        :param obj: Union[zarr.Array, zarr.Group]: The Zarr Data to encode
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: Always True, indicating that the encoding was successful
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name, encoded data string, data type, and object name.
        """
        obj_type = "Group" if isinstance(obj, zarr.Group) else "Array"
        obj_name = obj.name if isinstance(obj, zarr.Array) else None

        with tempfile.TemporaryDirectory() as tmpdirname:
            store_path = os.path.join(tmpdirname, "zarr_store")

            if obj_type == "Array":
                zarr.save_array(store_path, obj)
            else:
                zarr.save_group(store_path, obj)

            with io.BytesIO() as binary_stream:
                with zipfile.ZipFile(binary_stream, "w") as zipf:
                    for foldername, subfolders, filenames in os.walk(store_path):
                        for filename in filenames:
                            zipf.write(
                                os.path.join(foldername, filename),
                                arcname=os.path.relpath(
                                    os.path.join(foldername, filename), store_path
                                ),
                            )
                obj_data = base64.b64encode(binary_stream.getvalue()).decode("ascii")

        return True, dict(
            __wrapyfi__=(str(self.__class__.__name__), obj_data, obj_type, obj_name)
        )

    def decode(self, obj_type, obj_full, *args, **kwargs):
        """
        Decode a zip compressed and base64 encoded string back into Zarr Data.

        :param obj_type: type: The expected type of the decoded object (not used)
        :param obj_full: tuple: A tuple containing the encoded data string, data type, and object name
        :param args: tuple: Additional arguments (not used)
        :param kwargs: dict: Additional keyword arguments (not used)
        :return: Tuple[bool, Union[zarr.Array, zarr.Group]]: A tuple containing:
            - bool: Always True, indicating that the decoding was successful
            - Union[zarr.Array, zarr.Group]: The decoded Zarr Data
        """
        obj_data = obj_full[1]
        zarr_type = obj_full[2]
        zarr_name = obj_full[
            3
        ]  # zarr_name is used only if zarr_type is 'Array'. Currently not used.

        with io.BytesIO(base64.b64decode(obj_data.encode("ascii"))) as binary_stream:
            with tempfile.TemporaryDirectory() as tmpdirname:
                store_path = os.path.join(tmpdirname, "zarr_store")
                with zipfile.ZipFile(binary_stream, "r") as zipf:
                    zipf.extractall(path=store_path)

                if zarr_type == "Array":
                    array = zarr.open_array(store_path, mode="r")
                    return True, array
                else:
                    group = zarr.open_group(store_path, mode="r")
                    return True, group
