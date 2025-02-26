import os
from pathlib import Path
from glob import glob
import threading
import importlib.util
import sys
from typing import Union, Callable, Any, Optional, List


WRAPYFI_PLUGIN_PATHS = "WRAPYFI_PLUGIN_PATHS"
WRAPYFI_MWARE_PATHS = "WRAPYFI_MWARE_PATHS"

lock = threading.Lock()


def deepcopy(
    obj: Any,
    exclude_keys: Optional[Union[list, tuple]] = None,
    shallow_keys: Optional[Union[list, tuple]] = None,
):
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
            ret = {
                key: deepcopy(val, exclude_keys)
                for key, val in obj.items()
                if key not in exclude_keys + _shallows
            }
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


def match_args(
    args: Union[list, tuple],
    kwargs: dict,
    src_args: Union[list, tuple],
    src_kwargs: dict,
):
    """
    Match and Substitute Arguments and Keyword Arguments using Specified Source Values.

    Navigate through the provided `args` and `kwargs`, identifying entries prefixed with "$" and substituting
    them with values from `src_args` and `src_kwargs` respectively, to dynamically modify the function call
    parameters using the source values.

    :param args: Union[list, tuple]: A list of arguments, potentially containing strings that indicate substitutable entries.
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
    :return: Tuple[list, dict]: A tuple containing:
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
        elif (
            isinstance(kwarg_val, str)
            and "$" in kwarg_val
            and not kwarg_val[1:].isdigit()
        ):
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
            module = __import__(module_name, fromlist=["*"])
        except ImportError as e:
            # print(module_name + " could not be imported.", e)
            continue
        if hasattr(module, "__all__"):
            all_names = module.__all__
        else:
            all_names = [name for name in dir(module) if not name.startswith("_")]
        globals.update({name: getattr(module, name) for name in all_names})


def scan_external(module_paths: str, component: str):
    """
    Scan external directories specified in module_paths for a specified component directory
    and dynamically import the modules.

    Args:
    :param module_paths: str: Colon-separated paths for external directories to scan.
    :param component: str: Type of module to scan for i.e., "listeners", "publishers", "servers", "clients", "plugins"
    """
    # Split the provided paths and iterate through them
    extern_modules_paths = module_paths.split(":")

    for mod_group_idx, extern_module_path in enumerate(extern_modules_paths):
        # Define the directory to search based on the component type
        extern_base_dir = Path(extern_module_path) / component
        extern_modules = list(extern_base_dir.glob("*.py"))

        for mod_idx, extern_module in enumerate(extern_modules):
            # Generate a unique name for each module
            spec_name = (
                f"wrapyfi.extern{mod_group_idx}.{component}.{extern_module.stem}"
            )

            # Dynamically load and execute the module
            spec = importlib.util.spec_from_file_location(spec_name, extern_module)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Register the loaded module in the global scope
            dynamic_module_import([spec.name], globals())


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
                    cls._instances[cls] = super(SingletonOptimized, cls).__call__(
                        *args, **kwargs
                    )
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
        base_dir = Path(__file__).parent / "plugins"
        modules = glob(str(base_dir / "*.py"), recursive=True)

        modules = [
            "wrapyfi.plugins." + Path(module).relative_to(base_dir).as_posix()
            for module in modules
        ]
        dynamic_module_import(modules, globals())
        scan_external(os.environ.get(WRAPYFI_PLUGIN_PATHS, ""), "plugins")
