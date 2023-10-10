import os
from glob import glob
import threading
import importlib.util
import sys
from typing import Union, Callable, Any, Optional, List


WRAPYFI_PLUGIN_PATHS = "WRAPYFI_PLUGIN_PATHS"


lock = threading.Lock()


def deepcopy(obj: Any, exclude_keys: Optional[Union[list, tuple]] = None, shallow_keys: Optional[Union[list, tuple]] = None):
    """
    Deep copy an object, excluding specified keys.

    :param obj: Any: The object to deep copy
    :param exclude_keys: Union[list, tuple]: A list of keys to exclude from the deep copy
    :param shallow_keys: Union[list, tuple]: A list of keys to shallow copy
    """
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
            return copy.deepcopy(obj)


def get_default_args(fnc: Callable[..., Any]):
    """
    Get the default arguments for a function.

    :param fnc: Callable[..., Any]: The function to get the default arguments for
    """
    import inspect
    signature = inspect.signature(fnc)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def match_args(args: Union[list, tuple], kwargs: dict, src_args: Union[list, tuple], src_kwargs: dict):
    """
    Match and Substitute Arguments and Keyword Arguments using Specified Source Values.

    Navigate through the provided `args` and `kwargs`, identifying entries prefixed with "$" and substituting
    them with values from `src_args` and `src_kwargs` respectively, to dynamically modify the function call
    parameters using the source values.

    :param args: Union[list, tuple]:
        A list of arguments, potentially containing strings that indicate substitutable entries.
        Substitutable entries are prefixed with "$" and followed by either:
            - A digit (indicating an index to reference a value from `src_args`), or
            - Non-digit characters (indicating a key to reference a value from `src_kwargs`).
    :param kwargs: dict:
        A dictionary of keyword arguments, where values might be strings indicating substitutable entries,
        similar to the entries in `args`.
    :param src_args: Union[list, tuple]:
        A list of source arguments, intended to be referenced by substitutable entries within `args`.
    :param src_kwargs: dict:
        A dictionary of source keyword arguments, intended to be referenced by substitutable entries within `args`
        and `kwargs`.
    :return: Tuple[list, dict]:
        A tuple containing:
            - list: The new arguments, formed by substituting specified entries from `args` using `src_args` and `src_kwargs`.
            - dict: The new keyword arguments, formed by substituting specified entries from `kwargs` using `src_args` and `src_kwargs`.
    """
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


def dynamic_module_import(modules: List[str], globals: dict):
    """
    Dynamically import modules.

    :param modules: List[str]: A list of module names to import
    :param globals: dict: The globals dictionary to update
    """
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
    """
    A singleton metaclass that is thread-safe and optimized for speed.

    Source: https://stackoverflow.com/a/6798042
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonOptimized, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Plugin(object):
    """
    Base class for encoding and decoding plugins.
    """
    def encode(self, *args, **kwargs):
        """
        Encode data into a base64 string.

        :param args: tuple: Additional arguments
        :param kwargs: dict: Additional keyword arguments
        :return: Tuple[bool, dict]: A tuple containing:
            - bool: True if the encoding was successful, False otherwise
            - dict: A dictionary containing:
                - '__wrapyfi__': A tuple containing the class name and encoded data string
        """
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        """
        Decode a base64 string back into data.

        :param args: tuple: Additional arguments
        :param kwargs: dict: Additional keyword arguments
        :return: Tuple[bool, object]: A tuple containing:
            - bool: True if the decoding was successful, False otherwise
            - object: The decoded data
        """
        raise NotImplementedError


class PluginRegistrar(object):
    """
    Class for registering encoding and decoding plugins.
    """
    encoder_registry = {}
    decoder_registry = {}

    @staticmethod
    def register(types=None):
        """
        Register a plugin for encoding and decoding a specific type.

        :param types: tuple: The type(s) to register the plugin for
        """
        def wrapper(cls):
            if types is not None:
                for cls_type in types:
                    PluginRegistrar.encoder_registry[cls_type] = cls
                PluginRegistrar.decoder_registry[str(cls.__name__)] = cls
            return cls
        return wrapper

    @staticmethod
    def scan():
        """
        Scan the plugins directory (Wrapyfi builtin and external) for plugins to register.
        This method is called automatically when the module is imported.
        """
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


