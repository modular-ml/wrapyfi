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


@PluginRegistrar.register(types=None if not HAVE_ZARR else zarr.Array.__mro__[:-1] + zarr.Group.__mro__[:-1])
class ZarrData(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):
        obj_type = 'Group' if isinstance(obj, zarr.Group) else 'Array'
        obj_name = obj.name if isinstance(obj, zarr.Array) else None

        with tempfile.TemporaryDirectory() as tmpdirname:
            store_path = os.path.join(tmpdirname, 'zarr_store')

            if obj_type == 'Array':
                zarr.save_array(store_path, obj)
            else:
                zarr.save_group(store_path, obj)

            with io.BytesIO() as binary_stream:
                with zipfile.ZipFile(binary_stream, 'w') as zipf:
                    for foldername, subfolders, filenames in os.walk(store_path):
                        for filename in filenames:
                            zipf.write(os.path.join(foldername, filename),
                                       arcname=os.path.relpath(os.path.join(foldername, filename),
                                                               store_path))
                obj_data = base64.b64encode(binary_stream.getvalue()).decode('ascii')

        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_type, obj_name))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        obj_data = obj_full[1]
        zarr_type = obj_full[2]
        zarr_name = obj_full[3]  # zarr_name is used only if zarr_type is 'Array'. Currently not used.

        with io.BytesIO(base64.b64decode(obj_data.encode('ascii'))) as binary_stream:
            with tempfile.TemporaryDirectory() as tmpdirname:
                store_path = os.path.join(tmpdirname, 'zarr_store')
                with zipfile.ZipFile(binary_stream, 'r') as zipf:
                    zipf.extractall(path=store_path)

                if zarr_type == 'Array':
                    array = zarr.open_array(store_path, mode='r')
                    return True, array
                else:
                    group = zarr.open_group(store_path, mode='r')
                    return True, group
