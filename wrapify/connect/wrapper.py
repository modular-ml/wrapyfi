from functools import wraps
import inspect
import copy

import wrapify.connect.publishers as pub
import wrapify.connect.listeners as lsn
from wrapify.utils import get_default_args, match_args
from wrapify.config.manager import ConfigManager

DEFAULT_COMMUNICATOR = "yarp"


class MiddlewareCommunicator(object):
    __registry = {}

    def __init__(self):
        self.config = ConfigManager(None).config
        if self.__class__.__name__ in self.config:
            for key, value in self.config[self.__class__.__name__].items():
                self.activate_communication(key, mode=value)

    @classmethod
    def register(cls, *args, **kwargs):
        def encapsulate(fnc):
            # define the communication message type (single element)
            if isinstance(args[0], str):
                return_func_listen = lsn.Listeners.registry[args[0]+":"+args[1]]
                return_func_publish = pub.Publishers.registry[args[0]+":"+args[1]]
                return_func_args = args[2:]
                return_func_kwargs = kwargs
                return_func_type = args[0]+":"+args[1]

            # define the communication message type (list for single return). NOTE: supports 1 layer depth only
            elif isinstance(args[0], list):
                return_func_listen, return_func_publish, \
                return_func_args, return_func_kwargs, return_func_type = [], [], [], [], []
                for arg in args:
                    return_func_listen.append(lsn.Listeners.registry[arg[0]+":"+arg[1]])
                    return_func_publish.append(pub.Publishers.registry[arg[0]+":"+arg[1]])
                    return_func_args.append([a for a in arg[2:] if not isinstance(a, dict)])
                    return_func_kwargs.append(*[a for a in arg[2:] if isinstance(a, dict)])
                    return_func_type.append(arg[0]+":"+arg[1])

            # define the communication message type (dict for single return). NOTE: supports 1 layer depth only
            elif isinstance(args[0], dict):
                raise NotImplementedError("Dictionaries are not yet supported as a return type")

            if fnc.__qualname__ in cls.__registry:
                cls.__registry[fnc.__qualname__]["communicator"].append({
                    "return_func_listen": return_func_listen,
                    "return_func_publish": return_func_publish,
                    "return_func_args": return_func_args,
                    "return_func_kwargs": return_func_kwargs,
                    "return_func_type": return_func_type})
            else:
                cls.__registry[fnc.__qualname__] = {"communicator": [{
                    "return_func_listen": return_func_listen,
                    "return_func_publish": return_func_publish,
                    "return_func_args": return_func_args,
                    "return_func_kwargs": return_func_kwargs,
                    "return_func_type": return_func_type}]}
            cls.__registry[fnc.__qualname__]["mode"] = None

            @wraps(fnc)
            def wrapper(*wds, **kwds):
                # triggers on calling the function
                instance_address = str(wds).split(" ")[-1].rstrip(",)")
                instance_id = cls._MiddlewareCommunicator__registry[fnc.__qualname__]["__WRAPIFY_INSTANCES"].index(instance_address) + 1
                instance_id = "." + str(instance_id) if instance_id > 1 else ""

                kwd = get_default_args(fnc)
                kwd.update(kwds)
                cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["args"] = wds
                cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["kwargs"] = kwd

                # execute the function as usual
                if cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["mode"] is None:
                    return fnc(*wds, **kwds)

                # publishes the functions returns
                elif cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["mode"] == "publish":
                    if "wrapped_executor" not in cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["communicator"][0]:
                        cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["communicator"].reverse()
                        # instantiate the publishers
                        for communicator in cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["communicator"]:
                            # single element
                            if isinstance(communicator["return_func_type"], str):
                                new_args, new_kwargs = match_args(
                                    communicator["return_func_args"], communicator["return_func_kwargs"], wds[1:], kwd)
                                communicator["wrapped_executor"] = communicator["return_func_publish"](*new_args, **new_kwargs)
                            # list for single return
                            elif isinstance(communicator["return_func_type"], list):
                                communicator["wrapped_executor"] = []
                                for comm_idx in range(len(communicator["return_func_type"])):
                                    new_args, new_kwargs = match_args(
                                        communicator["return_func_args"][comm_idx], communicator["return_func_kwargs"][comm_idx], wds[1:], kwd)
                                    communicator["wrapped_executor"].append(communicator["return_func_publish"][comm_idx](*new_args, **new_kwargs))
                    returns = fnc(*wds, **kwds)
                    for ret_idx, ret in enumerate(returns):
                        wrp_exec = cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["communicator"][ret_idx]["wrapped_executor"]
                        # single element
                        if isinstance(wrp_exec, pub.Publisher):
                            wrp_exec.publish(ret)
                        # list for single return
                        elif isinstance(wrp_exec, list):
                            for wrp_idx, wrp in enumerate(wrp_exec):
                                wrp.publish(ret[wrp_idx])
                    return returns

                # listens to the publisher and returns the messages
                elif cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["mode"] == "listen":
                    if "wrapped_executor" not in cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["communicator"][0]:
                        cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["communicator"].reverse()
                        # instantiate the listeners
                        for communicator in cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["communicator"]:
                            # single element
                            if isinstance(communicator["return_func_type"], str):
                                new_args, new_kwargs = match_args(communicator["return_func_args"], communicator["return_func_kwargs"], wds[1:], kwd)
                                communicator["wrapped_executor"] = communicator["return_func_listen"](*new_args, **new_kwargs)
                            # list for single return
                            elif isinstance(communicator["return_func_type"], list):
                                communicator["wrapped_executor"] = []
                                for comm_idx in range(len(communicator["return_func_type"])):
                                    new_args, new_kwargs = match_args(communicator["return_func_args"][comm_idx], communicator["return_func_kwargs"][comm_idx], wds[1:], kwd)
                                    communicator["wrapped_executor"].append(communicator["return_func_listen"][comm_idx](*new_args, **new_kwargs))
                    cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["last_results"] = []
                    for ret_idx in range(len(cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["communicator"])):
                        wrp_exec = cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["communicator"][ret_idx]["wrapped_executor"]
                        # single element
                        if isinstance(wrp_exec, lsn.Listener):
                            cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["last_results"].append(wrp_exec.listen())
                            # list for single return
                        elif isinstance(wrp_exec, list):
                            ret = []
                            for wrp_idx, wrp in enumerate(wrp_exec):
                                ret.append(wrp.listen())
                            cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["last_results"].append(ret)

                    return cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["last_results"]

                # WARNING: use with caution. This produces "None" for all the function's returns
                elif  cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["mode"] == "disable":
                    cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["last_results"] = []
                    for ret_idx in range(len(cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["communicator"])):
                        cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["last_results"].append(None)
                    return cls._MiddlewareCommunicator__registry[fnc.__qualname__+instance_id]["last_results"]

            return wrapper
        return encapsulate

    def activate_communication(self, fnc_name, mode="publish"):
        instance_addr = str(self).split(" ")[-1]
        if self.__class__.__name__+"."+fnc_name in self.__registry:
            if "__WRAPIFY_INSTANCES" in self.__registry[self.__class__.__name__+"."+fnc_name]:
                if not instance_addr in self.__registry[self.__class__.__name__+"."+fnc_name]["__WRAPIFY_INSTANCES"]:
                    self.__registry[self.__class__.__name__ + "." + fnc_name]["__WRAPIFY_INSTANCES"].append(instance_addr)
                    new_instance = True
                else:
                    new_instance = False
            else:
                self.__registry[self.__class__.__name__ + "." + fnc_name]["__WRAPIFY_INSTANCES"] = [instance_addr]
                new_instance = True
            if new_instance:
                instance_id = len(self.__registry[self.__class__.__name__ + "." + fnc_name]["__WRAPIFY_INSTANCES"])
                if instance_id > 1:
                    instance_id = "." + str(instance_id)
                    self.__registry[self.__class__.__name__ + "." + fnc_name + instance_id] = \
                        copy.deepcopy(self.__registry[self.__class__.__name__ + "." + fnc_name])
                else:
                    instance_id = ""
            else:
                instance_id = self.__registry[self.__class__.__name__]["__WRAPIFY_INSTANCES"].index(instance_addr) + 1
                instance_id = "." + str(instance_id) if instance_id > 1 else ""

            self.__registry[self.__class__.__name__+"."+fnc_name + instance_id]["mode"] = mode



