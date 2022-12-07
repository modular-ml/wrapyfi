import logging
import atexit
import threading
import multiprocessing

import zmq

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator

ZEROMQ_POST_OPTS = ["SUBSCRIBE", "UNSUBSCRIBE", "LINGER", "ROUTER_HANDOVER", "ROUTER_MANDATORY", "PROBE_ROUTER", "XPUB_VERBOSE", "XPUB_VERBOSER", "REQ_CORRELATE", "REQ_RELAXED", "SNDHWM", "RCVHWM"]

class ZeroMQMiddlewarePubSub(metaclass=SingletonOptimized):

    @staticmethod
    def activate(**kwargs):
        zeromq_post_kwargs = {}
        zeromq_pre_kwargs = {}
        for key, value in kwargs.items():
            try:
                getattr(zmq, key)
                if key in ZEROMQ_POST_OPTS:
                    zeromq_post_kwargs[key] = value
                else:
                    zeromq_pre_kwargs[key] = value
            except AttributeError:
                pass

        ZeroMQMiddlewarePubSub(zeromq_proxy_kwargs=kwargs, zeromq_post_kwargs=zeromq_post_kwargs, **zeromq_pre_kwargs)

    def __init__(self, zeromq_proxy_kwargs=None, zeromq_post_kwargs=None, **kwargs):
        self.zeromq_proxy_kwargs = zeromq_proxy_kwargs or {}
        self.zeromq_kwargs = zeromq_post_kwargs or {}
        logging.info("Initialising ZeroMQ PUB/SUB middleware")
        ctx = zmq.Context.instance()
        for socket_property in kwargs.items():
            if isinstance(socket_property[1], str):
                ctx.setsockopt_string(getattr(zmq, socket_property[0]), socket_property[1])
            else:
                ctx.setsockopt(getattr(zmq, socket_property[0]), socket_property[1])
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

        if zeromq_proxy_kwargs is not None and zeromq_proxy_kwargs:
            if zeromq_proxy_kwargs["proxy_broker_spawn"] == "process":
                proxy = multiprocessing.Process(name='zeromq_broker', target=self.__init_proxy, kwargs=kwargs)
                proxy.daemon = True
                proxy.start()
            else:  # if threaded
                proxy = threading.Thread(name='zeromq_broker', target=self.__init_proxy, kwargs=kwargs)
                proxy.setDaemon(True)
                proxy.start()
            pass

    @staticmethod
    def __init_proxy(socket_address="tcp://*:5555", socket_sub_address="tcp://*:5556", **kwargs):
        xpub = zmq.Context.instance().socket(zmq.XPUB)
        try:
            xpub.bind(socket_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_address}")
            return
        xsub = zmq.Context.instance().socket(zmq.XSUB)
        try:
            xsub.bind(socket_sub_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_sub_address}")
            return
        logging.info(f"[ZeroMQ] Intialising PUB/SUB proxy broker")
        zmq.proxy(xpub, xsub)


    @staticmethod
    def deinit():
        logging.info("Deinitialising ZeroMQ middleware")
        zmq.Context.instance().destroy()


class ZeroMQMiddlewareRepReq(metaclass=SingletonOptimized):

    @staticmethod
    def activate(**kwargs):
        zmq_kwargs = {}
        for key, value in kwargs.items():
            try:
                getattr(zmq, key)
                zmq_kwargs[key] = value
            except AttributeError:
                pass

        ZeroMQMiddlewareRepReq(zmq_proxy_kwargs=kwargs)
        return zmq_kwargs

    def __init__(self, zmq_proxy_kwargs=None, *args, **kwargs):
        logging.info("Initialising ZeroMQ REP/REQ middleware")
        ctx = zmq.Context.instance()
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

        if zmq_proxy_kwargs is not None and zmq_proxy_kwargs:
            if zmq_proxy_kwargs["proxy_broker_spawn"] == "process":
                proxy = multiprocessing.Process(name='zeromq_broker', target=self.__init_device, kwargs=zmq_proxy_kwargs)
                proxy.daemon = True
                proxy.start()
            else:  # if threaded
                proxy = threading.Thread(name='zeromq_broker', target=self.__init_device, kwargs=zmq_proxy_kwargs)
                proxy.setDaemon(True)
                proxy.start()
            pass

    @staticmethod
    def __init_device(socket_address="tcp://*:5559", socket_sub_address="tcp://*:5560", **kwargs):
        xrep = zmq.Context.instance().socket(zmq.XREP)
        try:
            xrep.bind(socket_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_address}")
            return
        xreq = zmq.Context.instance().socket(zmq.XREQ)
        try:
            xreq.bind(socket_sub_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_sub_address}")
            return
        logging.info(f"[ZeroMQ] Intialising REP/REQ device broker")
        zmq.proxy(xrep, xreq)

    @staticmethod
    def deinit():
        logging.info("Deinitialising ZeroMQ middleware")
        zmq.Context.instance().destroy()
