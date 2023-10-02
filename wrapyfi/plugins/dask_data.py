import io
import json
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


@PluginRegistrar.register(types=None if not HAVE_DASK else dd.DataFrame.__mro__[:-1] + da.Array.__mro__[:-1])
class DaskData(Plugin):
    def __init__(self, **kwargs):
        pass

    def encode(self, obj, *args, **kwargs):
        obj_type = None
        with io.BytesIO() as memfile:
            if isinstance(obj, dd.DataFrame):
                pandas_df = obj.compute().reset_index()
                memfile.write(pandas_df.to_json(orient="records").encode('ascii'))
                obj_type = 'DataFrame'
            elif isinstance(obj, da.Array):
                np.save(memfile, obj.compute(), allow_pickle=True)
                obj_type = 'Array'
            memfile.seek(0)
            obj_data = base64.b64encode(memfile.read()).decode('ascii')
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_type))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        obj_data = base64.b64decode(obj_full[1].encode('ascii'))
        obj_type = obj_full[2]
        with io.BytesIO(obj_data) as memfile:
            if obj_type == 'DataFrame':
                pandas_df = pd.read_json(memfile.read().decode('ascii'), orient="records")
                pandas_df.set_index('index', inplace=True)  # Set the index back after reading from JSON
                return True, dd.from_pandas(pandas_df, npartitions=2)
            elif obj_type == 'Array':
                np_array = np.load(memfile, allow_pickle=True)
                return True, da.from_array(np_array, chunks=np_array.shape)