import os
import asyncio
import time
from functools import wraps, partial
import re
from typing import Union, List, Any, Callable, Optional

import wrapyfi.connect.publishers as pub
import wrapyfi.connect.listeners as lsn
import wrapyfi.connect.servers as srv
import wrapyfi.connect.clients as clt

from wrapyfi.utils import get_default_args, match_args, deepcopy
from wrapyfi.config.manager import ConfigManager


DEFAULT_COMMUNICATOR = os.environ.get("WRAPYFI_DEFAULT_COMMUNICATOR", "zeromq")
DEFAULT_COMMUNICATOR = os.environ.get("WRAPYFI_DEFAULT_MWARE", DEFAULT_COMMUNICATOR)


class MiddlewareCommunicator(object):
    __registry = {}

    def __init__(self):
        """
        Initialises the middleware communicator.
        """
        self.config = ConfigManager(None).config
        if self.__class__.__name__ in self.config:
            for key, value in self.config[self.__class__.__name__].items():
                self.activate_communication(getattr(self.__class__, key), mode=value)

    @classmethod
    def __trigger_publish(
        cls, func: Callable[..., Any], instance_id: str, kwd: dict, *wds, **kwds
    ):
        """
        Triggers the publish mode of the middleware communicator. If the intended publisher type and middleware are unavailable, resorts to a fallback publisher instead.

        :param func: Callable[..., Any]: The function to be triggered and whose return values are to be published
        :param instance_id: str: The unique identifier of the function instance, utilized to access the middleware communicator registry and manage different instances of communications
        :param kwd: dict: A dictionary containing keyword arguments to be matched and potentially utilized during the publisher instantiation and function invocation
        :param wds: tuple: Variable positional arguments to be passed to the function
        :param kwds: dict: Additional keyword arguments to be passed to the function
        :return: Any: The return values of the triggered function (`func`). The data type and structure depend on the output of `func`
        """
        if (
            "wrapped_executor"
            not in cls._MiddlewareCommunicator__registry[
                func.__qualname__ + instance_id
            ]["communicator"][0]
        ):
            # instantiate the publishers
            cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id][
                "communicator"
            ].reverse()
            for communicator in cls._MiddlewareCommunicator__registry[
                func.__qualname__ + instance_id
            ]["communicator"]:
                # single element
                if isinstance(communicator["return_func_type"], str):
                    return_func_pub_kwargs = deepcopy(
                        communicator["return_func_kwargs"]
                    )
                    return_func_pub_kwargs.update(
                        return_func_pub_kwargs.get("publisher_kwargs", {})
                    )
                    return_func_pub_kwargs.pop("listener_kwargs", None)
                    return_func_pub_kwargs.pop("publisher_kwargs", None)
                    new_args, new_kwargs = match_args(
                        communicator["return_func_args"],
                        return_func_pub_kwargs,
                        wds[1:],
                        kwd,
                    )
                    return_func_type = communicator["return_func_type"]
                    return_func_middleware = new_kwargs.pop(
                        "middleware", DEFAULT_COMMUNICATOR
                    )
                    try:
                        communicator["wrapped_executor"] = pub.Publishers.registry[
                            return_func_type + return_func_middleware
                        ](*new_args, **new_kwargs)
                    except KeyError:
                        communicator["wrapped_executor"] = pub.Publishers.registry[
                            "MMO:fallback"
                        ](
                            *new_args,
                            missing_middleware_object=return_func_type
                            + return_func_middleware,
                            **new_kwargs,
                        )
                        communicator["return_func_type"] = "MMO:"
                # list for single return
                elif isinstance(communicator["return_func_type"], list):
                    communicator["wrapped_executor"] = []
                    for comm_idx in range(len(communicator["return_func_type"])):
                        return_func_pub_kwargs = deepcopy(
                            communicator["return_func_kwargs"][comm_idx]
                        )
                        return_func_pub_kwargs.update(
                            return_func_pub_kwargs.get("publisher_kwargs", {})
                        )
                        return_func_pub_kwargs.pop("listener_kwargs", None)
                        return_func_pub_kwargs.pop("publisher_kwargs", None)
                        new_args, new_kwargs = match_args(
                            communicator["return_func_args"][comm_idx],
                            return_func_pub_kwargs,
                            wds[1:],
                            kwd,
                        )
                        return_func_type = communicator["return_func_type"][comm_idx]
                        return_func_middleware = new_kwargs.pop(
                            "middleware", DEFAULT_COMMUNICATOR
                        )
                        try:
                            communicator["wrapped_executor"].append(
                                pub.Publishers.registry[
                                    return_func_type + return_func_middleware
                                ](*new_args, **new_kwargs)
                            )
                        except KeyError:
                            communicator["wrapped_executor"].append(
                                pub.Publishers.registry["MMO:fallback"](
                                    *new_args,
                                    missing_middleware_object=return_func_type
                                    + return_func_middleware,
                                    **new_kwargs,
                                )
                            )
                            communicator["return_func_type"][comm_idx] = "MMO:"

        returns = func(*wds, **kwds)
        for ret_idx, ret in enumerate(returns):
            wrp_exec = cls._MiddlewareCommunicator__registry[
                func.__qualname__ + instance_id
            ]["communicator"][ret_idx]["wrapped_executor"]
            # single element
            if isinstance(wrp_exec, pub.Publisher):
                wrp_exec.publish(ret)
            # list for single return
            elif isinstance(wrp_exec, list):
                for wrp_idx, wrp in enumerate(wrp_exec):
                    wrp.publish(ret[wrp_idx])
        return returns

    @classmethod
    def __trigger_listen(
        cls, func: Callable[..., Any], instance_id: str, kwd: dict, *wds, **kwds
    ):
        """
        Triggers the listen mode of the middleware communicator. If the intended listener type and middleware are unavailable, resorts to a fallback listener instead.

        :param func: Callable[..., Any]: The function associated with the listener and whose return values might be utilized
        :param instance_id: str: The unique identifier of the function instance, utilized to access the middleware communicator registry and manage different instances of communications
        :param kwd: dict: A dictionary containing keyword arguments to be matched and potentially utilized during the listener instantiation and function invocation
        :param wds: tuple: Variable positional arguments to be passed to the function
        :param kwds: dict: Additional keyword arguments to be passed to the function (not used since the listener does not accept any additional arguments, given that it does not execute any function or transmit any arguments)
        :return: Any: The return values obtained from the listeners. The data type and structure depend on the output of the listener
        """
        if (
            "wrapped_executor"
            not in cls._MiddlewareCommunicator__registry[
                func.__qualname__ + instance_id
            ]["communicator"][0]
        ):
            # instantiate the listeners
            cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id][
                "communicator"
            ].reverse()
            for communicator in cls._MiddlewareCommunicator__registry[
                func.__qualname__ + instance_id
            ]["communicator"]:
                # single element
                if isinstance(communicator["return_func_type"], str):
                    return_func_lsn_kwargs = deepcopy(
                        communicator["return_func_kwargs"]
                    )
                    return_func_lsn_kwargs.update(
                        return_func_lsn_kwargs.get("listener_kwargs", {})
                    )
                    return_func_lsn_kwargs.pop("listener_kwargs", None)
                    return_func_lsn_kwargs.pop("publisher_kwargs", None)
                    new_args, new_kwargs = match_args(
                        communicator["return_func_args"],
                        return_func_lsn_kwargs,
                        wds[1:],
                        kwd,
                    )
                    return_func_type = communicator["return_func_type"]
                    return_func_middleware = new_kwargs.pop(
                        "middleware", DEFAULT_COMMUNICATOR
                    )
                    try:
                        communicator["wrapped_executor"] = lsn.Listeners.registry[
                            return_func_type + return_func_middleware
                        ](*new_args, **new_kwargs)
                    except KeyError:
                        communicator["wrapped_executor"] = lsn.Listeners.registry[
                            "MMO:fallback"
                        ](
                            *new_args,
                            missing_middleware_object=return_func_type
                            + return_func_middleware,
                            **new_kwargs,
                        )
                        communicator["return_func_type"] = "MMO:"

                # list for single return
                elif isinstance(communicator["return_func_type"], list):
                    communicator["wrapped_executor"] = []
                    for comm_idx in range(len(communicator["return_func_type"])):
                        return_func_lsn_kwargs = deepcopy(
                            communicator["return_func_kwargs"][comm_idx]
                        )
                        return_func_lsn_kwargs.update(
                            return_func_lsn_kwargs.get("listener_kwargs", {})
                        )
                        return_func_lsn_kwargs.pop("listener_kwargs", None)
                        return_func_lsn_kwargs.pop("publisher_kwargs", None)
                        new_args, new_kwargs = match_args(
                            communicator["return_func_args"][comm_idx],
                            return_func_lsn_kwargs,
                            wds[1:],
                            kwd,
                        )
                        return_func_type = communicator["return_func_type"][comm_idx]
                        return_func_middleware = new_kwargs.pop(
                            "middleware", DEFAULT_COMMUNICATOR
                        )
                        try:
                            communicator["wrapped_executor"].append(
                                lsn.Listeners.registry[
                                    return_func_type + return_func_middleware
                                ](*new_args, **new_kwargs)
                            )
                        except KeyError:
                            communicator["wrapped_executor"].append(
                                lsn.Listeners.registry["MMO:fallback"](
                                    *new_args,
                                    missing_middleware_object=return_func_type
                                    + return_func_middleware,
                                    **new_kwargs,
                                )
                            )
                            communicator["return_func_type"][comm_idx] = "MMO:"

        returns = []
        for ret_idx in range(
            len(
                cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id][
                    "communicator"
                ]
            )
        ):
            wrp_exec = cls._MiddlewareCommunicator__registry[
                func.__qualname__ + instance_id
            ]["communicator"][ret_idx]["wrapped_executor"]
            # single element
            if isinstance(wrp_exec, lsn.Listener):
                returns.append(wrp_exec.listen())
                # list for single return
            elif isinstance(wrp_exec, list):
                subreturns = []
                for wrp_idx, wrp in enumerate(wrp_exec):
                    subreturns.append(wrp.listen())
                returns.append(subreturns)
        return returns

    @classmethod
    async def __async_trigger_publish(
        cls, func: Callable[..., Any], instance_id: str, publish_kwargs: dict, *wds, **kwds
    ):
        """
        Asynchronous wrapper for the synchronous publish method.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, partial(cls.__trigger_publish, func, instance_id, publish_kwargs, *wds, **kwds)
        )

    @classmethod
    async def __async_trigger_listen(
        cls, func: Callable[..., Any], instance_id: str, listen_kwargs: dict, *wds, **kwds
    ):
        """
        Asynchronous wrapper for the synchronous listen method.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(cls.__trigger_listen, func, instance_id, listen_kwargs, *wds, **kwds)
        )


    @classmethod
    async def __async_trigger_transceive(
        cls, func: Callable[..., Any], instance_id: str, kwd: dict, *wds, **kwds
    ):
        """
        Triggers the transceive mode of the middleware communicator. Asynchronous transceive method that runs publish and listen concurrently. The returns are acquired from the listener.
        """
        publish_kwargs = deepcopy(kwd.get("publish_kwargs", {}))
        listen_kwargs = deepcopy(kwd.get("listen_kwargs", {}))

        if not publish_kwargs:
            publish_kwargs.update(kwd)
        if not listen_kwargs:
            listen_kwargs.update(kwd)

        # run publish and listen concurrently
        publish_task = cls.__async_trigger_publish(func, instance_id, publish_kwargs, *wds, **kwds)
        listen_task = cls.__async_trigger_listen(func, instance_id + ":rec", listen_kwargs, *wds, **kwds)

        # await tasks and return listened output
        _, listened_output = await asyncio.gather(publish_task, listen_task)
        return listened_output

    @classmethod
    async def __async_trigger_reemit(
        cls, func: Callable[..., Any], instance_id: str, kwd: dict, *wds, **kwds
    ):
        """
        Triggers the reemit mode of the middleware communicator. Asynchronous reemit method that runs listen, takes the returns and publishes them. The returns arrive from the publisher.
        """
        publish_kwargs = deepcopy(kwd.get("publish_kwargs", {}))
        listen_kwargs = deepcopy(kwd.get("listen_kwargs", {}))

        if not publish_kwargs:
            publish_kwargs.update(kwd)
        if not listen_kwargs:
            listen_kwargs.update(kwd)

        # await tasks and get listened output
        listen_task = cls.__async_trigger_listen(func, instance_id + ":rec", listen_kwargs, *wds, **kwds)
        listened_output = await asyncio.gather(listen_task)

        # receive listened outputs, run method, and publish the output
        publish_task = cls.__async_trigger_publish(func, instance_id, publish_kwargs, *listened_output, **kwds)
        published_output = await asyncio.gather(publish_task)

        return published_output

    @classmethod
    def __trigger_reply(
        cls, func: Callable[..., Any], instance_id: str, kwd, *wds, **kwds
    ):
        """
        Triggers the reply mode of the middleware communicator. If the intended server type and middleware are unavailable, resorts to a fallback server instead.

        :param func: Callable[..., Any]: The function to be triggered and whose return values may be used in replies
        :param instance_id: str: The unique identifier of the function instance, utilized to access the middleware communicator registry and manage different instances of communications
        :param kwd: dict: A dictionary containing keyword arguments to be matched and potentially utilized during the server instantiation and function invocation
        :param wds: tuple: Variable positional arguments to be passed to the function
        :param kwds: dict: Additional keyword arguments to be passed to the function
        :return: Any: The return values of the triggered function (`func`). The data type and structure depend on the output of `func`
        """
        if (
            "wrapped_executor"
            not in cls._MiddlewareCommunicator__registry[
                func.__qualname__ + instance_id
            ]["communicator"][0]
        ):
            # instantiate the publishers
            cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id][
                "communicator"
            ].reverse()
            for communicator in cls._MiddlewareCommunicator__registry[
                func.__qualname__ + instance_id
            ]["communicator"]:
                # single element
                if isinstance(communicator["return_func_type"], str):
                    return_func_pub_kwargs = deepcopy(
                        communicator["return_func_kwargs"]
                    )
                    return_func_pub_kwargs.update(
                        return_func_pub_kwargs.get("publisher_kwargs", {})
                    )
                    return_func_pub_kwargs.pop("listener_kwargs", None)
                    return_func_pub_kwargs.pop("publisher_kwargs", None)
                    new_args, new_kwargs = match_args(
                        communicator["return_func_args"],
                        return_func_pub_kwargs,
                        wds[1:],
                        kwd,
                    )
                    return_func_type = communicator["return_func_type"]
                    return_func_middleware = new_kwargs.pop(
                        "middleware", DEFAULT_COMMUNICATOR
                    )
                    try:
                        communicator["wrapped_executor"] = srv.Servers.registry[
                            return_func_type + return_func_middleware
                        ](*new_args, **new_kwargs)
                    except KeyError:
                        communicator["wrapped_executor"] = srv.Servers.registry[
                            "MMO:fallback"
                        ](
                            *new_args,
                            missing_middleware_object=return_func_type
                            + return_func_middleware,
                            **new_kwargs,
                        )
                        communicator["return_func_type"] = "MMO:"

                # list for single return
                elif isinstance(communicator["return_func_type"], list):
                    communicator["wrapped_executor"] = []
                    for comm_idx in range(len(communicator["return_func_type"])):
                        return_func_pub_kwargs = deepcopy(
                            communicator["return_func_kwargs"][comm_idx]
                        )
                        return_func_pub_kwargs.update(
                            return_func_pub_kwargs.get("publisher_kwargs", {})
                        )
                        return_func_pub_kwargs.pop("listener_kwargs", None)
                        return_func_pub_kwargs.pop("publisher_kwargs", None)
                        new_args, new_kwargs = match_args(
                            communicator["return_func_args"][comm_idx],
                            return_func_pub_kwargs,
                            wds[1:],
                            kwd,
                        )
                        return_func_type = communicator["return_func_type"][comm_idx]
                        return_func_middleware = new_kwargs.pop(
                            "middleware", DEFAULT_COMMUNICATOR
                        )
                        try:
                            communicator["wrapped_executor"].append(
                                srv.Servers.registry[
                                    return_func_type + return_func_middleware
                                ](*new_args, **new_kwargs)
                            )
                        except KeyError:
                            communicator["wrapped_executor"].append(
                                srv.Servers.registry["MMO:fallback"](
                                    *new_args,
                                    missing_middleware_object=return_func_type
                                    + return_func_middleware,
                                    **new_kwargs,
                                )
                            )
                            communicator["return_func_type"][comm_idx] = "MMO:"

        returns = None
        for ret_idx, functor in enumerate(
            cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id][
                "communicator"
            ]
        ):
            wrp_exec = functor["wrapped_executor"]
            # single element
            if isinstance(wrp_exec, srv.Server):
                new_args, new_kwargs = wrp_exec.await_request(*wds[1:], **kwds)
                if returns is None:
                    returns = func(wds[0], *new_args, **new_kwargs)
                ret = returns[ret_idx]
                wrp_exec.reply(ret)
            # list for single return
            elif isinstance(wrp_exec, list):
                for wrp_idx, wrp in enumerate(wrp_exec):
                    new_args, new_kwargs = wrp.await_request(*wds[1:], **kwds)
                    if returns is None:
                        returns = func(wds[0], *new_args, **new_kwargs)
                    ret = returns[ret_idx][wrp_idx]
                    wrp.reply(ret)
        return returns

    @classmethod
    def __trigger_request(
        cls, func: Callable[..., Any], instance_id: str, kwd, *wds, **kwds
    ):
        """
        Triggers the request mode of the middleware communicator. If the intended client type and middleware are unavailable, resorts to a fallback client instead.

        :param func: Callable[..., Any]: The function associated with the client requests and might utilize the request results
        :param instance_id: str: The unique identifier of the function instance, utilized to access the middleware communicator registry and manage different instances of communications
        :param kwd: dict: A dictionary containing keyword arguments to be matched and potentially utilized during the client instantiation and function invocation
        :param wds: tuple: Variable positional arguments to be passed to the function
        :param kwds: dict: Additional keyword arguments to be passed to the function
        :return: Any: The return values obtained from the clients' requests. The data type and structure depend on the output of the client requests
        """
        if (
            "wrapped_executor"
            not in cls._MiddlewareCommunicator__registry[
                func.__qualname__ + instance_id
            ]["communicator"][0]
        ):
            # instantiate the listeners
            cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id][
                "communicator"
            ].reverse()
            for communicator in cls._MiddlewareCommunicator__registry[
                func.__qualname__ + instance_id
            ]["communicator"]:
                # single element
                if isinstance(communicator["return_func_type"], str):
                    return_func_lsn_kwargs = deepcopy(
                        communicator["return_func_kwargs"]
                    )
                    return_func_lsn_kwargs.update(
                        return_func_lsn_kwargs.get("listener_kwargs", {})
                    )
                    return_func_lsn_kwargs.pop("listener_kwargs", None)
                    return_func_lsn_kwargs.pop("publisher_kwargs", None)
                    new_args, new_kwargs = match_args(
                        communicator["return_func_args"],
                        return_func_lsn_kwargs,
                        wds[1:],
                        kwd,
                    )
                    return_func_type = communicator["return_func_type"]
                    return_func_middleware = new_kwargs.pop(
                        "middleware", DEFAULT_COMMUNICATOR
                    )
                    try:
                        communicator["wrapped_executor"] = clt.Clients.registry[
                            return_func_type + return_func_middleware
                        ](*new_args, **new_kwargs)
                    except KeyError:
                        communicator["wrapped_executor"] = clt.Clients.registry[
                            "MMO:fallback"
                        ](
                            *new_args,
                            missing_middleware_object=return_func_type
                            + return_func_middleware,
                            **new_kwargs,
                        )
                        communicator["return_func_type"] = "MMO:"

                # list for single return
                elif isinstance(communicator["return_func_type"], list):
                    communicator["wrapped_executor"] = []
                    for comm_idx in range(len(communicator["return_func_type"])):
                        return_func_lsn_kwargs = deepcopy(
                            communicator["return_func_kwargs"][comm_idx]
                        )
                        return_func_lsn_kwargs.update(
                            return_func_lsn_kwargs.get("listener_kwargs", {})
                        )
                        return_func_lsn_kwargs.pop("listener_kwargs", None)
                        return_func_lsn_kwargs.pop("publisher_kwargs", None)
                        new_args, new_kwargs = match_args(
                            communicator["return_func_args"][comm_idx],
                            return_func_lsn_kwargs,
                            wds[1:],
                            kwd,
                        )
                        return_func_type = communicator["return_func_type"][comm_idx]
                        return_func_middleware = new_kwargs.pop(
                            "middleware", DEFAULT_COMMUNICATOR
                        )
                        try:
                            communicator["wrapped_executor"].append(
                                clt.Clients.registry[
                                    return_func_type + return_func_middleware
                                ](*new_args, **new_kwargs)
                            )
                        except KeyError:
                            communicator["wrapped_executor"].append(
                                clt.Clients.registry["MMO:fallback"](
                                    *new_args,
                                    missing_middleware_object=return_func_type
                                    + return_func_middleware,
                                    **new_kwargs,
                                )
                            )
                            communicator["return_func_type"][comm_idx] = "MMO:"

        returns = []
        for ret_idx, functor in enumerate(
            cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id][
                "communicator"
            ]
        ):
            wrp_exec = functor["wrapped_executor"]
            # single element
            if isinstance(wrp_exec, clt.Client):
                ret = wrp_exec.request(*wds[1:], **kwds)
                returns.append(ret)
            # list for single return
            elif isinstance(wrp_exec, list):
                subreturns = []
                for wrp_idx, wrp in enumerate(wrp_exec):
                    ret = wrp.request(*wds[1:], **kwds)
                    subreturns.append(ret[wrp_idx])
                returns.append(subreturns)
        return returns

    @classmethod
    def register(
        cls,
        data_type: Union[str, List[Any]],
        middleware: str = DEFAULT_COMMUNICATOR,
        *args,
        **kwargs,
    ):
        """
        Registers a function to the middleware communicator, defining its communication message type and associated
        middleware. Note that the function returned is a wrapper that can alter the behavior of the registered function
        based on the communication mode set in the middleware communicator registry.

        :param data_type: Union[str, List[Any]]: Specifies the communication message type, either as a string or a list containing specifications of the data type and middleware
        :param middleware: str: Specifies the middleware to be used for the communication, defaults to DEFAULT_COMMUNICATOR
        :param args: tuple: Variable positional arguments to be passed to the function
        :param kwargs: dict: Additional keyword arguments to be passed to the function
        :return: Callable[..., Any]: A wrapper function that triggers specific communication modes (e.g., publish, listen, reply, request) based on the registered function and middleware settings

        :raises: NotImplementedError: If `data_type` is a dictionary or an unsupported type
        """

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
                    return_func_args.append(
                        [a for a in arg[2:] if not isinstance(a, dict)]
                    )
                    return_func_kwargs.append(
                        *[a for a in arg[2:] if isinstance(a, dict)]
                    )
                    return_func_kwargs[-1]["middleware"] = str(arg[1])
                    return_func_type.append(data_spec)

            # define the communication message type (dict for single return). NOTE: supports 1 layer depth only
            elif isinstance(data_type, dict):
                raise NotImplementedError(
                    "Dictionaries are not yet supported as a return type"
                )

            else:
                raise NotImplementedError(
                    f"Return data type not supported: {data_type}"
                )

            func_qualname = func.__qualname__
            if func_qualname in cls.__registry:
                cls.__registry[func_qualname]["communicator"].append(
                    {
                        "return_func_args": return_func_args,
                        "return_func_kwargs": return_func_kwargs,
                        "return_func_type": return_func_type,
                    }
                )
            else:
                cls.__registry[func_qualname] = {
                    "communicator": [
                        {
                            "return_func_args": return_func_args,
                            "return_func_kwargs": return_func_kwargs,
                            "return_func_type": return_func_type,
                        }
                    ]
                }
            cls.__registry[func_qualname]["mode"] = None

            @wraps(func)
            def wrapper(*wds, **kwds):  # triggers on calling the method
                if hasattr(func, "__wrapped__"):
                    return func(*wds, **kwds)

                instance_address = hex(id(wds[0]))
                try:
                    instance_id = (
                        cls._MiddlewareCommunicator__registry[func.__qualname__][
                            "__WRAPYFI_INSTANCES"
                        ].index(instance_address)
                        + 1
                    )
                    instance_id = "" if instance_id <= 1 else "." + str(instance_id)
                except KeyError:
                    instance_id = ""

                # execute the method as usual
                if (
                    cls._MiddlewareCommunicator__registry[
                        func.__qualname__ + instance_id
                    ]["mode"]
                    is None
                ):
                    return func(*wds, **kwds)

                kwd = get_default_args(func)
                kwd.update(kwds)
                cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id][
                    "args"
                ] = wds
                cls._MiddlewareCommunicator__registry[func.__qualname__ + instance_id][
                    "kwargs"
                ] = kwd

                # publishes the method returns
                if (
                    cls._MiddlewareCommunicator__registry[
                        func.__qualname__ + instance_id
                    ]["mode"]
                    == "publish"
                ):
                    return cls.__trigger_publish(func, instance_id, kwd, *wds, **kwds)

                # listens to the publisher and returns the messages
                elif (
                    cls._MiddlewareCommunicator__registry[
                        func.__qualname__ + instance_id
                    ]["mode"]
                    == "listen"
                ):
                    return cls.__trigger_listen(func, instance_id, kwd, *wds, **kwds)

                # publishes the method returns and listens for another message, which is eventually returned
                elif (
                        cls._MiddlewareCommunicator__registry[
                            func.__qualname__ + instance_id
                        ]["mode"]
                        == "transceive"
                ):
                    return asyncio.run(cls.__async_trigger_transceive(func, instance_id, kwd, *wds, **kwds))

                # listens for a message, invokes the method and publishes the returns
                elif (
                        cls._MiddlewareCommunicator__registry[
                            func.__qualname__ + instance_id
                            ]["mode"]
                        == "reemit"
                ):
                    return asyncio.run(cls.__async_trigger_reemit(func, instance_id, kwd, *wds, **kwds))

                # server awaits request from client and replies with method returns
                elif (
                    cls._MiddlewareCommunicator__registry[
                        func.__qualname__ + instance_id
                    ]["mode"]
                    == "reply"
                ):
                    return cls.__trigger_reply(func, instance_id, kwd, *wds, **kwds)

                # client requests with args from server and awaits reply
                elif (
                    cls._MiddlewareCommunicator__registry[
                        func.__qualname__ + instance_id
                    ]["mode"]
                    == "request"
                ):
                    return cls.__trigger_request(func, instance_id, kwd, *wds, **kwds)

                # WARNING: use with caution. This produces "None" for all the method's returns
                elif (
                    cls._MiddlewareCommunicator__registry[
                        func.__qualname__ + instance_id
                    ]["mode"]
                    == "disable"
                ):
                    cls._MiddlewareCommunicator__registry[
                        func.__qualname__ + instance_id
                    ]["last_results"] = []
                    for ret_idx in range(
                        len(
                            cls._MiddlewareCommunicator__registry[
                                func.__qualname__ + instance_id
                            ]["communicator"]
                        )
                    ):
                        cls._MiddlewareCommunicator__registry[
                            func.__qualname__ + instance_id
                        ]["last_results"].append(None)
                    return cls._MiddlewareCommunicator__registry[
                        func.__qualname__ + instance_id
                    ]["last_results"]

            return wrapper

        return encapsulate

    def activate_communication(
        self, func: Union[str, Callable[..., Any]], mode: Union[str, List[str]], id_postfix: Optional[str] = False
    ):
        """
        Activates the communication mode for a registered function in the middleware communicator.
        The mode determines how the function will interact with the middleware communicator upon invocation,
        e.g., whether it should publish its return values, listen for values, etc.
        The communication mode can be set for all instances of the function or individually per instance.

        :param func: Union[str, Callable[..., Any]]: The function or the name of the function for which the communication mode should be activated
        :param mode: Union[str, List[str]]: Specifies the communication mode to be activated, either as a single mode (string) applicable to all instances, or as a list of modes (strings) per instance
        :param id_postfix: Optional[str]: Specifies postfix to method registry name. This is needed for tranceive and reemit modes

        :raises: IndexError: If mode is a list and the number of elements does not match the number of instances
        :raises: ValueError: If the instance address cannot be found in the registry
        """
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
                        self.__registry[f"{func.__qualname__}.{instance_id}"] = (
                            deepcopy(entry, exclude_keys=["wrapped_executor"])
                        )

            instance_qualname = (
                f"{func.__qualname__}.{instance_id}"
                if instance_id > 1
                else func.__qualname__
            )
            instance_qualname += id_postfix if id_postfix else ""

            if isinstance(mode, list):
                try:
                    current_mode = mode[instance_id - 1]
                    self.__registry[instance_qualname]["mode"] = current_mode
                except IndexError:
                    raise IndexError(
                        "When mode (publish|listen|transceive|reemit|disable|null) specified in configuration file is a "
                        "list, No. of elements in the list should match the number of instances"
                    )
            else:
                current_mode = mode
                self.__registry[instance_qualname]["mode"] = current_mode
            self.__registry[instance_qualname]["instance_addr"] = instance_addr

            if current_mode == "transceive":
                self.__registry[instance_qualname + ":rec"] = (
                    deepcopy(entry)
                )
                self.activate_communication(func, mode="transceive:rec", id_postfix=":rec")
            elif current_mode == "reemit":
                self.__registry[instance_qualname + ":rec"] = (
                    deepcopy(entry)
                )
                self.activate_communication(func, mode="reemit:rec", id_postfix=":rec")

    def close(self):
        """
        Closes this middleware communicator instance.
        """
        self.close_instance(hex(id(self)))

    @staticmethod
    def get_communicators():
        """
        Returns the available middleware communicators.

        :return: Set[str]: The available middleware communicators (excluding the fallback communicator)
        """
        return (pub.Publishers.mwares | lsn.Listeners.mwares | srv.Servers.mwares | clt.Clients.mwares) - {"fallback"}

    @classmethod
    def close_all_instances(cls):
        """
        Closes all instances of the middleware communicator.
        """
        close_instance_idxs = set()
        for val in cls._MiddlewareCommunicator__registry.values():
            instance_addr = val.get("instance_addr", False)
            close_instance_idxs.add(instance_addr) if instance_addr else None
        for close_instance_idx in close_instance_idxs:
            cls.close_instance(close_instance_idx)

    @classmethod
    def close_instance(cls, instance_addr: Optional[str] = None):
        """
        Closes a specific instance of the middleware communicator. Note that the instance address is the hexadecimal
        representation of the instance's id. If no instance address is provided, all instances will be closed.

        :param instance_addr: str: The unique identifier of the instance to be closed, defaults to None

        :raises: ValueError: If the instance address cannot be found in the registry
        """
        while True:
            del_entry = False
            del_entry_name = None
            del_entry_idx = None
            other_entry_keys = []
            # find self in registry and remove id from instances
            for entry_key, entry_val in cls._MiddlewareCommunicator__registry.items():
                if instance_addr in entry_val.get("__WRAPYFI_INSTANCES", []):
                    if not del_entry:
                        del_entry_idx = entry_val["__WRAPYFI_INSTANCES"].index(
                            instance_addr
                        )
                        if del_entry_idx == 0:
                            del_entry = entry_key
                        else:
                            del_entry = (
                                re.sub("\.\d+", "\.", entry_key)
                                + "."
                                + str(del_entry_idx + 1)
                            )
                        del_entry_name = re.sub("\.\d+", "\.", entry_key)
                    else:
                        if del_entry_name in entry_key:
                            other_entry_keys.append(entry_key)
                    entry_val["__WRAPYFI_INSTANCES"].remove(instance_addr)

            # delete registry entry and all its publishers/listeners/servers/clients
            if del_entry:
                for communicator in cls._MiddlewareCommunicator__registry[del_entry][
                    "communicator"
                ]:
                    wrapped_executor = communicator.get("wrapped_executor", False)
                    if wrapped_executor:
                        if isinstance(wrapped_executor, list):
                            for single_wrapped_executor in wrapped_executor:
                                single_wrapped_executor.close()
                        else:
                            wrapped_executor.close()
                        communicator.pop("wrapped_executor")
                if other_entry_keys:
                    del cls._MiddlewareCommunicator__registry[del_entry]
                else:
                    cls._MiddlewareCommunicator__registry[del_entry].pop(
                        "__WRAPYFI_INSTANCES"
                    )
                    cls._MiddlewareCommunicator__registry[del_entry].pop("mode")
                    cls._MiddlewareCommunicator__registry[del_entry].pop(
                        "instance_addr"
                    )

                # shift all entries backwards following the deleted one
                if del_entry_idx - 1 < len(other_entry_keys):
                    for other_entry_key in other_entry_keys[del_entry_idx:]:
                        new_key = re.split("\.(\d+)", other_entry_key)
                        if len(new_key) == 1:
                            new_key = new_key[0]
                        elif str(int(new_key[1]) - 1) == "1":
                            new_key = new_key[0]
                        else:
                            new_key = new_key[0] + "." + str(int(new_key[1]) - 1)
                        cls._MiddlewareCommunicator__registry[new_key] = (
                            cls._MiddlewareCommunicator__registry.pop(other_entry_key)
                        )
            else:
                break

    def __del__(self):
        self.close()
