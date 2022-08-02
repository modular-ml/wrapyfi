import io
import json
import base64
import threading
import numpy as np

lock = threading.Lock()


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
            new_kwargs[kwarg_key] = kwarg_val
    return tuple(new_args), new_kwargs


def dynamic_module_import(modules, globals):
    for module_name in modules:
        if not module_name.endswith(".py") or module_name.endswith("__.py"):
            continue
        module_name = module_name[:-3]
        module_name = module_name.replace("/", ".")
        module = __import__(module_name, fromlist=['*'])
        if hasattr(module, '__all__'):
            all_names = module.__all__
        else:
            all_names = [name for name in dir(module) if not name.startswith('_')]
        globals.update({name: getattr(module, name) for name in all_names})


class SingletonOptimized(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonOptimized, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class JsonEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.ndarray):
            with io.BytesIO() as memfile:
                np.save(memfile, obj)
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            return dict(__wrapify__=('numpy.ndarray', obj_data))

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

class JsonDecodeHook:

    def __init__(self, torch_device=None):
        self.torch_device = torch_device

    def object_hook(self, obj):

        if isinstance(obj, dict):
            wrapify = obj.get('__wrapify__', None)
            if wrapify is not None:
                obj_type = wrapify[0]
                if obj_type == 'numpy.ndarray':
                    with io.BytesIO(base64.b64decode(wrapify[1].encode('ascii'))) as memfile:
                        return np.load(memfile)

        return obj
