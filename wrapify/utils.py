
def get_default_args(fnc):
    import inspect
    signature = inspect.signature(fnc)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def match_args(args, kwargs, src_args, src_kwargs):
    new_args = []
    new_kwargs = {}
    for arg in args:
        if arg[0] == "$" and arg[1:].isdigit():
            new_args.append(src_args[int(arg[1:])])
        elif arg[0] == "$" and not arg[1:].isdigit():
            new_args.append(src_kwargs[arg[1:]])
        else:
            new_args.append(arg)

    for kwarg_key, kwarg_val in kwargs.items():
        if isinstance(kwarg_val, str) and "$" in kwarg_val and kwarg_val[1:].isdigit():
            new_kwargs[kwarg_key] = src_args[int(kwarg_val[1:])]
        elif isinstance(kwarg_val, str) and "$" in kwarg_val and not kwarg_val[1:].isdigit():
            new_kwargs[kwarg_key] = src_kwargs[kwarg_val[1:]]
        else:
            new_kwargs[kwarg_key] =  kwarg_val
    return tuple(new_args), new_kwargs


# code adapted from: https://stackoverflow.com/a/27948073
import json


class JsonEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(JsonEncoder, self).__init__(*args, **kwargs)

    def default(self, obj):
        """
        if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
        """
        import base64
        import numpy as np
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(obj.data).decode()
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def json_numpy_obj_hook(dct):
        """
        Decodes a previously encoded numpy ndarray
        with proper shape and dtype
        :param dct: (dict) json encoded ndarray
        :return: (ndarray) if input was an encoded ndarray
        """
        import base64
        import numpy as np

        if isinstance(dct, dict) and '__ndarray__' in dct:
            data = base64.b64decode(dct['__ndarray__'])
            return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
        return dct

    # Overload dump/load to default use this behavior.
    @staticmethod
    def dumps(*args, **kwargs):
        kwargs.setdefault('cls', JsonEncoder)
        return json.dumps(*args, **kwargs)

    @staticmethod
    def loads(*args, **kwargs):
        kwargs.setdefault('object_hook', JsonEncoder.json_numpy_obj_hook)
        return json.loads(*args, **kwargs)

    @staticmethod
    def dump(*args, **kwargs):
        kwargs.setdefault('cls', JsonEncoder)
        return json.dump(*args, **kwargs)

    @staticmethod
    def load(*args, **kwargs):
        kwargs.setdefault('object_hook', JsonEncoder.json_numpy_obj_hook)
        return json.load(*args, **kwargs)