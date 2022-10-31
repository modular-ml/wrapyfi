import os
from glob import glob
import threading
import importlib.util
import sys


WRAPYFI_PLUGIN_PATHS = "WRAPYFI_PLUGIN_PATHS"


lock = threading.Lock()


def deepcopy(obj, exclude_keys=None, shallow_keys=None):
    import copy
    if exclude_keys is None:
        return copy.deepcopy(obj)
    else:
        if isinstance(obj, list):
            return [deepcopy(item, exclude_keys) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(deepcopy(item, exclude_keys) for item in obj)
        elif isinstance(obj, set):
            return {deepcopy(item, exclude_keys) for item in obj}
        elif isinstance(obj, dict):
            _shallows = shallow_keys or []
            ret = {key: deepcopy(val, exclude_keys) for key, val in obj.items() if key not in exclude_keys + _shallows}
            ret.update({key: val for key, val in obj.items() if key in _shallows})
            return ret
        else:
            return obj


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
        try:
            module = __import__(module_name, fromlist=['*'])
        except ImportError:
            continue
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


class Plugin(object):
    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError


class PluginRegistrar(object):
    encoder_registry = {}
    decoder_registry = {}

    @staticmethod
    def register(types=None):
        def wrapper(cls):
            if types is not None:
                for cls_type in types:
                    PluginRegistrar.encoder_registry[cls_type] = cls
                PluginRegistrar.decoder_registry[str(cls.__name__)] = cls
            return cls
        return wrapper

    @staticmethod
    def scan():
        # TODO (fabawi): also scan plugin directories in os env WRAPYFI_PLUGIN_PATHS
        modules = glob(os.path.join(os.path.dirname(__file__), "plugins", "*.py"), recursive=True)
        modules = ["wrapyfi.plugins." + module.replace(os.path.dirname(__file__) + "/plugins/", "") for module in modules]
        dynamic_module_import(modules, globals())

        extern_modules_paths = os.environ.get(WRAPYFI_PLUGIN_PATHS, "").split(":")
        for mod_group_idx, extern_module_path in enumerate(extern_modules_paths):
            extern_modules = glob(os.path.join(extern_module_path, "plugins", "*.py"), recursive=True)
            for mod_idx, extern_module in enumerate(extern_modules):
                spec = importlib.util.spec_from_file_location(f"wrapyfi.extern{mod_group_idx}.plugins{mod_idx}", extern_module)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                extern_modules = [f"wrapyfi.extern{mod_group_idx}.plugins{mod_idx}." + extern_module.replace(
                    extern_module_path + "/plugins/", "")]
                dynamic_module_import(extern_modules, globals())


