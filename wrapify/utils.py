import os
from glob import glob
import threading


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
    registry = {}

    @staticmethod
    def register(cls):
        PluginRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        # TODO (fabawi): also scan plugin directories in os env WRAPIFY_PLUGIN_PATHS
        modules = glob(os.path.join(os.path.dirname(__file__), "plugins", "*.py"), recursive=True)
        modules = ["wrapify.plugins." + module.replace(os.path.dirname(__file__) + "/plugins/", "") for module in modules]
        dynamic_module_import(modules, globals())

