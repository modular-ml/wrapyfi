import os
import copy
from functools import wraps

import wrapyfi.connect.publishers as pub
import wrapyfi.connect.listeners as lsn
from wrapyfi.utils import get_default_args, match_args
from wrapyfi.config.manager import ConfigManager


DEFAULT_COMMUNICATOR = os.environ.get("WRAPYFI_DEFAULT_COMMUNICATOR", "zeromq")
DEFAULT_COMMUNICATOR = os.environ.get("WRAPYFI_DEFAULT_MWARE", DEFAULT_COMMUNICATOR)

class MiddlewareCommunicator(object):
    __registry = {}

    def __init__(self):
        self.config = ConfigManager(None).config
        if self.__class__.__name__ in self.config:
            for key, value in self.config[self.__class__.__name__].items():
                self.activate_communication(getattr(self.__class__, key), mode=value)

    @classmethod
    def register(cls, data_type, middleware=DEFAULT_COMMUNICATOR, *args, **kwargs):
        def encapsulate(func):
            # define the communication message type (single element)
            if isinstance(data_type, str):
                return_func_type = data_type + ":"
                return_func_args = args
                return_func_kwargs = kwargs
                return_func_kwargs["middleware"] = str(middleware)

            # define the communication message type (list for single return). NOTE: supports 1 layer depth only
            elif isinstance(data_type, list):
                return_func_args, return_func_kwargs, return_func_type = [], [], []
                for arg in data_type:
                    data_spec = arg[0] + ":"
                    return_func_args.append([a for a in arg[2:] if not isinstance(a, dict)])
                    return_func_kwargs.append(*[a for a in arg[2:] if isinstance(a, dict)])
                    return_func_kwargs[-1]["middleware"] = str(arg[1])
                    return_func_type.append(data_spec)

            # define the communication message type (dict for single return). NOTE: supports 1 layer depth only
            elif isinstance(data_type, dict):
                raise NotImplementedError("Dictionaries are not yet supported as a return type")

            else:
                raise NotImplementedError(f"Return data type not supported: {data_type}")

            func_qualname = func.__qualname__
            if func_qualname in cls.__registry:
                cls.__registry[func_qualname]["communicator"].append({
                    "return_func_args": return_func_args,
                    "return_func_kwargs": return_func_kwargs,
                    "return_func_type": return_func_type})
            else:
                cls.__registry[func_qualname] = {"communicator": [{
                    "return_func_args": return_func_args,
                    "return_func_kwargs": return_func_kwargs,
                    "return_func_type": return_func_type}]}
            cls.__registry[func_qualname]["mode"] = None

            @wraps(func)
            def wrapper(*wds, **kwds):  # Triggers on calling the function
                if hasattr(func, "__wrapped__"):
                    return func(*wds, **kwds)

                # execute the function as usual
                if cls._MiddlewareCommunicator__registry[func.__qualname__]["mode"] is None:
                    return func(*wds, **kwds)

                instance_address = hex(id(wds[0]))
                instance_id = cls._MiddlewareCommunicator__registry[func.__qualname__]["__WRAPYFI_INSTANCES"].index(instance_address) + 1
                instance_id = "." + str(instance_id) if instance_id > 1 else ""

                kwd = get_default_args(func)
                kwd.update(kwds)
                cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["args"] = wds
                cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["kwargs"] = kwd

                # publishes the functions returns
                if cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["mode"] == "publish":
                    if "wrapped_executor" not in cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["communicator"][0]:
                        # instantiate the publishers
                        cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["communicator"].reverse()
                        for communicator in cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["communicator"]:
                            # single element
                            if isinstance(communicator["return_func_type"], str):
                                return_func_pub_kwargs = copy.deepcopy(communicator["return_func_kwargs"])
                                return_func_pub_kwargs.update(return_func_pub_kwargs.get("publisher_kwargs", {}))
                                return_func_pub_kwargs.pop("listener_kwargs", None)
                                return_func_pub_kwargs.pop("publisher_kwargs", None)
                                new_args, new_kwargs = match_args(
                                    communicator["return_func_args"], return_func_pub_kwargs, wds[1:], kwd)
                                return_func_type = communicator["return_func_type"]
                                return_func_middleware = new_kwargs.pop("middleware", DEFAULT_COMMUNICATOR)
                                communicator["wrapped_executor"] = pub.Publishers.registry[return_func_type + return_func_middleware](*new_args, **new_kwargs)
                            # list for single return
                            elif isinstance(communicator["return_func_type"], list):
                                communicator["wrapped_executor"] = []
                                for comm_idx in range(len(communicator["return_func_type"])):
                                    return_func_pub_kwargs = copy.deepcopy(communicator["return_func_kwargs"][comm_idx])
                                    return_func_pub_kwargs.update(return_func_pub_kwargs.get("publisher_kwargs", {}))
                                    return_func_pub_kwargs.pop("listener_kwargs", None)
                                    return_func_pub_kwargs.pop("publisher_kwargs", None)
                                    new_args, new_kwargs = match_args(
                                        communicator["return_func_args"][comm_idx], return_func_pub_kwargs, wds[1:], kwd)
                                    return_func_type = communicator["return_func_type"][comm_idx]
                                    return_func_middleware = new_kwargs.pop("middleware", DEFAULT_COMMUNICATOR)
                                    communicator["wrapped_executor"].append(
                                        pub.Publishers.registry[return_func_type + return_func_middleware](*new_args, **new_kwargs))

                    returns = func(*wds, **kwds)
                    for ret_idx, ret in enumerate(returns):
                        wrp_exec = cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["communicator"][ret_idx]["wrapped_executor"]
                        # single element
                        if isinstance(wrp_exec, pub.Publisher):
                            wrp_exec.publish(ret)
                        # list for single return
                        elif isinstance(wrp_exec, list):
                            for wrp_idx, wrp in enumerate(wrp_exec):
                                wrp.publish(ret[wrp_idx])
                    return returns

                # listens to the publisher and returns the messages
                elif cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["mode"] == "listen":
                    if "wrapped_executor" not in cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["communicator"][0]:
                        # instantiate the listeners
                        cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["communicator"].reverse()
                        for communicator in cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["communicator"]:
                            # single element
                            if isinstance(communicator["return_func_type"], str):
                                return_func_lsn_kwargs = copy.deepcopy(communicator["return_func_kwargs"])
                                return_func_lsn_kwargs.update(return_func_lsn_kwargs.get("listener_kwargs", {}))
                                return_func_lsn_kwargs.pop("listener_kwargs", None)
                                return_func_lsn_kwargs.pop("publisher_kwargs", None)
                                new_args, new_kwargs = match_args(communicator["return_func_args"], return_func_lsn_kwargs, wds[1:], kwd)
                                return_func_type = communicator["return_func_type"]
                                return_func_middleware = new_kwargs.pop("middleware", DEFAULT_COMMUNICATOR)
                                communicator["wrapped_executor"] = lsn.Listeners.registry[return_func_type + return_func_middleware](*new_args, **new_kwargs)
                            # list for single return
                            elif isinstance(communicator["return_func_type"], list):
                                communicator["wrapped_executor"] = []
                                for comm_idx in range(len(communicator["return_func_type"])):
                                    return_func_lsn_kwargs = copy.deepcopy(communicator["return_func_kwargs"][comm_idx])
                                    return_func_lsn_kwargs.update(return_func_lsn_kwargs.get("listener_kwargs", {}))
                                    return_func_lsn_kwargs.pop("listener_kwargs", None)
                                    return_func_lsn_kwargs.pop("publisher_kwargs", None)
                                    new_args, new_kwargs = match_args(communicator["return_func_args"][comm_idx], return_func_lsn_kwargs, wds[1:], kwd)
                                    return_func_type = communicator["return_func_type"][comm_idx]
                                    return_func_middleware = new_kwargs.pop("middleware", DEFAULT_COMMUNICATOR)
                                    communicator["wrapped_executor"].append(
                                        lsn.Listeners.registry[return_func_type + return_func_middleware](*new_args, **new_kwargs))
                    cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["last_results"] = []
                    for ret_idx in range(len(cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["communicator"])):
                        wrp_exec = cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["communicator"][ret_idx]["wrapped_executor"]
                        # single element
                        if isinstance(wrp_exec, lsn.Listener):
                            cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["last_results"].append(wrp_exec.listen())
                            # list for single return
                        elif isinstance(wrp_exec, list):
                            ret = []
                            for wrp_idx, wrp in enumerate(wrp_exec):
                                ret.append(wrp.listen())
                            cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["last_results"].append(ret)

                    return cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["last_results"]

                # WARNING: use with caution. This produces "None" for all the function's returns
                elif cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["mode"] == "disable":
                    cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["last_results"] = []
                    for ret_idx in range(len(cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["communicator"])):
                        cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["last_results"].append(None)
                    return cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id]["last_results"]

            return wrapper
        return encapsulate

    def activate_communication(self, func, mode):
        if isinstance(func, str):
            func = getattr(self, func)
        entry = self.__registry.get(func.__qualname__, None)
        if entry is not None:
            instance_addr = hex(id(self))
            wrapyfi_instances = entry.get("__WRAPYFI_INSTANCES", None)
            if wrapyfi_instances is None:
                entry["__WRAPYFI_INSTANCES"] = [instance_addr]
                instance_id = 1
            else:
                try:
                    instance_id = wrapyfi_instances.index(instance_addr) + 1
                except ValueError:
                    wrapyfi_instances.append(instance_addr)
                    instance_id = len(wrapyfi_instances)
                    if instance_id > 1:
                        self.__registry[f"{func.__qualname__}.{instance_id}"] = copy.deepcopy(entry)
            instance_qualname = f"{func.__qualname__}.{instance_id}" if instance_id > 1 else func.__qualname__
            self.__registry[instance_qualname]["mode"] = mode

    @staticmethod
    def get_communicators():
        return pub.Publishers.mwares | lsn.Listeners.mwares
