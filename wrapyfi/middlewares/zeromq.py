import logging
import atexit
import threading
import multiprocessing
import time

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
        self.ctx = zmq.Context.instance()
        for socket_property in kwargs.items():
            if isinstance(socket_property[1], str):
                self.ctx.setsockopt_string(getattr(zmq, socket_property[0]), socket_property[1])
            else:
                self.ctx.setsockopt(getattr(zmq, socket_property[0]), socket_property[1])
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

        if zeromq_proxy_kwargs is not None and zeromq_proxy_kwargs:
            if zeromq_proxy_kwargs["proxy_broker_spawn"] == "process":
                self.proxy = multiprocessing.Process(name='zeromq_pubsub_broker', target=self.__init_proxy, kwargs=kwargs)
                self.proxy.daemon = True
                self.proxy.start()
            else:  # if threaded
                self.proxy = threading.Thread(name='zeromq_pubsub_broker', target=self.__init_proxy, kwargs=kwargs)
                self.proxy.setDaemon(True)
                self.proxy.start()
            pass

    @staticmethod
    def __init_proxy(socket_pub_address="tcp://127.0.0.1:5555", socket_sub_address="tcp://127.0.0.1:5556", **kwargs):
        xpub = zmq.Context.instance().socket(zmq.XPUB)
        try:
            xpub.bind(socket_pub_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_pub_address}")
            return
        xsub = zmq.Context.instance().socket(zmq.XSUB)
        try:
            xsub.bind(socket_sub_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_sub_address}")
            return
        # logging.info(f"[ZeroMQ] Intialising PUB/SUB proxy broker")
        zmq.proxy(xpub, xsub)


    @staticmethod
    def deinit():
        logging.info("Deinitialising ZeroMQ middleware")
        zmq.Context.instance().destroy()


class ZeroMQMiddlewareRepReq(metaclass=SingletonOptimized):

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

        ZeroMQMiddlewareRepReq(zeromq_proxy_kwargs=kwargs, zeromq_post_kwargs=zeromq_post_kwargs, **zeromq_pre_kwargs)

    def __init__(self, zeromq_proxy_kwargs=None, zeromq_post_kwargs=None, *args, **kwargs):
        self.zeromq_proxy_kwargs = zeromq_proxy_kwargs or {}
        self.zeromq_kwargs = zeromq_post_kwargs or {}
        logging.info("Initialising ZeroMQ REP/REQ middleware")
        self.ctx = zmq.Context.instance()
        for socket_property in kwargs.items():
            if isinstance(socket_property[1], str):
                self.ctx.setsockopt_string(getattr(zmq, socket_property[0]), socket_property[1])
            else:
                self.ctx.setsockopt(getattr(zmq, socket_property[0]), socket_property[1])
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

        if zeromq_proxy_kwargs is not None and zeromq_proxy_kwargs:
            if zeromq_proxy_kwargs["proxy_broker_spawn"] == "process":
                self.proxy = multiprocessing.Process(name='zeromq_repreq_broker', target=self.__init_device, kwargs=zeromq_proxy_kwargs)
                self.proxy.daemon = True
                self.proxy.start()
            else:  # if threaded
                self.proxy = threading.Thread(name='zeromq_repreq_broker', target=self.__init_device, kwargs=zeromq_proxy_kwargs)
                self.proxy.setDaemon(True)
                self.proxy.start()
            pass

    @staticmethod
    def __init_device(socket_rep_address="tcp://127.0.0.1:5559", server_req_address="tcp://127.0.0.1:5560", **kwargs):
        xrep = zmq.Context.instance().socket(zmq.XREP)
        try:
            xrep.bind(socket_rep_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_rep_address}")
            return
        xreq = zmq.Context.instance().socket(zmq.XREQ)
        try:
            xreq.bind(server_req_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {server_req_address}")
            return
        # logging.info(f"[ZeroMQ] Intialising REP/REQ device broker")
        zmq.proxy(xrep, xreq)

    @staticmethod
    def deinit():
        logging.info("Deinitialising ZeroMQ middleware")
        zmq.Context.instance().destroy()


class ZeroMQMiddlewareParamServer(metaclass=SingletonOptimized):

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

        ZeroMQMiddlewareParamServer(zeromq_proxy_kwargs=kwargs, zeromq_post_kwargs=zeromq_post_kwargs, **zeromq_pre_kwargs)

    def __init__(self, zeromq_proxy_kwargs=None, zeromq_post_kwargs=None, *args, **kwargs):
        self.zeromq_proxy_kwargs = zeromq_proxy_kwargs or {}
        self.zeromq_kwargs = zeromq_post_kwargs or {}
        logging.info("Initialising ZeroMQ Parameter Server")
        self.ctx = zmq.Context.instance()
        for socket_property in kwargs.items():
            if isinstance(socket_property[1], str):
                self.ctx.setsockopt_string(getattr(zmq, socket_property[0]), socket_property[1])
            else:
                self.ctx.setsockopt(getattr(zmq, socket_property[0]), socket_property[1])

        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

        if zeromq_proxy_kwargs is not None and zeromq_proxy_kwargs:
            self.manager = multiprocessing.Manager()
            self.params = self.manager.dict()
            self.params["WRAPYFI_ACTIVE"] = "True"
            if zeromq_proxy_kwargs["proxy_broker_spawn"] == "process":
                self.param_broadcaster = multiprocessing.Process(name='zeromq_param_broadcaster', target=self.__init_broadcaster,
                                                            kwargs=zeromq_proxy_kwargs, args=(self.params,))
                self.param_broadcaster.daemon = True
                self.param_broadcaster.start()
                self.param_server = multiprocessing.Process(name='zeromq_param_server', target=self.__init_server,
                                                            kwargs=zeromq_proxy_kwargs, args=(self.params,))
                self.param_server.daemon = True
                self.param_server.start()
            else:  # if threaded

                self.param_broadcaster = threading.Thread(name='zeromq_param_broadcaster', target=self.__init_broadcaster,
                                                     kwargs=zeromq_proxy_kwargs, args=(self.params,))
                self.param_broadcaster.setDaemon(True)
                self.param_broadcaster.start()
                self.param_server = threading.Thread(name='zeromq_param_server', target=self.__init_server,
                                                     kwargs=zeromq_proxy_kwargs, args=(self.params,))
                self.param_server.setDaemon(True)
                self.param_server.start()
            pass

    @staticmethod
    def __init_broadcaster(params, param_pub_address="tcp://127.0.0.1:5655", param_sub_address="tcp://127.0.0.1:5656",
                           param_poll_interval=1, verbose=False, **kwargs):
        update_trigger = False
        cached_params = {}
        root_topics = set()

        ctx = zmq.Context.instance()
        # create XPUB
        xpub_socket = ctx.socket(zmq.XPUB)
        xpub_socket.bind(param_pub_address)
        # create XSUB
        xsub_socket = ctx.socket(zmq.XSUB)
        xsub_socket.bind(param_sub_address)
        # connect a PUB socket to send parameters
        param_server = ctx.socket(zmq.PUB)
        param_server.connect(param_sub_address)
        # create poller
        poller = zmq.Poller()
        poller.register(xpub_socket, zmq.POLLIN)
        poller.register(xsub_socket, zmq.POLLIN)
        if verbose:
            logging.info(f"[ZeroMQ] Intialising PUB/SUB device broker")
        while True:
            # get event
            event = dict(poller.poll(param_poll_interval))
            if xpub_socket in event:
                message = xpub_socket.recv_multipart()
                if verbose:
                    logging.info("[ZeroMQ BROKER] xpub_socket recv message: %r" % message)
                if message[0].startswith(b"\x00"):
                    root_topics.remove(message[0][1:].decode("utf-8"))
                elif message[0].startswith(b"\x01"):
                    root_topics.add(message[0][1:].decode("utf-8"))
                xsub_socket.send_multipart(message)

            if xsub_socket in event:
                message = xsub_socket.recv_multipart()
                if verbose:
                    logging.info("[ZeroMQ BROKER] xsub_socket recv message: %r" % message)
                if message[0].startswith(b"\x01") or message[0].startswith(b"\x00"):
                    xpub_socket.send_multipart(message)
                else:
                    fltr_key = message[0].decode("utf-8")
                    fltr_message = {key: val for key, val in params.items()
                                    if key.startswith(fltr_key)}
                    if verbose:
                        logging.info("[ZeroMQ BROKER] xsub_socket filtered message: %r" % fltr_message)
                    for key, val in fltr_message.items():
                        prefix, param = key.rsplit("/", 1) if "/" in key else ("", key)
                        xpub_socket.send_multipart([prefix.encode("utf-8"), param.encode("utf-8"), val.encode("utf-8")])
                # xpub_socket.send_multipart(message)

            if event:
                update_trigger = True

            if param_server is not None:
                update_trigger, cached_params = ZeroMQMiddlewareParamServer.publish_params(
                    param_server, params, cached_params, root_topics, update_trigger)

    @staticmethod
    def publish_params(param_server, params, cached_params, root_topics, update_trigger):
        # Check if there are any subscribed clients before publishing updates
        if not root_topics:
            return None, None

        if not any((update_trigger, root_topics)) and cached_params == params:
            return None, None
        else:
            time.sleep(0.01)
            update_trigger = False
            cached_params = params.copy()

        # Publish updates for all parameters to subscribed clients
        for key, val in params.items():
            prefix, param = key.rsplit("/", 1) if "/" in key else ("", key)
            param_server.send_multipart([prefix.encode("utf-8"), param.encode("utf-8"), val.encode("utf-8")])

        return update_trigger, cached_params

    @staticmethod
    def __init_server(params, param_repreq_address="tcp://127.0.0.1:5659", **kwargs):
        ctx = zmq.Context.instance()
        request_server = ctx.socket(zmq.REP)
        request_server.bind(param_repreq_address)

        while True:
            # Receive requests from clients and handle them accordingly
            request = request_server.recv_string()
            if request.startswith("get"):
                try:
                    # Extract the parameter name and namespace prefix from the request
                    prefix, param = request[4:].rsplit("/", 1) if "/" in request[4:] else ("", request[4:])
                    # Construct the full parameter name with the namespace prefix
                    full_param = "/".join([prefix, param]) if prefix else param
                    if full_param in params:
                        request_server.send_string(str(params[full_param]))
                    else:
                        # Send an error message if the parameter does not exist
                        request_server.send_string("error:::parameter does not exist")
                except ValueError:
                    # Send an error message if the request is malformed
                    request_server.send_string("error:::malformed request")
            elif request.startswith("read"):
                try:
                    # Extract the parameter name and namespace prefix from the request
                    prefix = request[5:]
                    # Construct the full parameter name with the namespace prefix
                    if any(param.startswith(prefix) for param in params.keys()):
                        request_server.send_string(f"success:::{prefix}")
                    else:
                        # Send an error message if the parameter does not exist
                        request_server.send_string("error:::parameter does not exist")
                except ValueError:
                    # Send an error message if the request is malformed
                    request_server.send_string("error:::malformed request")
            elif request.startswith("set"):
                try:
                    # Extract the parameter name, namespace prefix and value from the request
                    prefix, param, value = request[4:].rsplit("/", 2)
                    # Construct the full parameter name with the namespace prefix
                    full_param = "/".join([prefix, param]) if prefix else param
                    params[full_param] = value
                    request_server.send_string(f"success:::{prefix}")
                except ValueError:
                    # Send an error message if the request is malformed
                    request_server.send_string("error:::malformed request")
            elif request.startswith("delete"):
                try:
                    # Extract the parameter name and namespace prefix from the request
                    prefix, param = request[7:].rsplit("/", 1)
                    # Construct the full parameter name with the namespace prefix
                    full_param = "/".join([prefix, param]) if prefix else param
                    if full_param in params:
                        del params[full_param]
                        request_server.send_string(f"success:::{prefix}")
                    else:
                        # Send an error message if the parameter does not exist
                        request_server.send_string("error:::parameter does not exist")
                except ValueError:
                    # Send an error message if the request is malformed
                    request_server.send_string("error:::malformed request")
            else:
                request_server.send_string("error:::invalid request")

    @staticmethod
    def deinit():
        logging.info("Deinitialising ZeroMQ Parameter Server")
        zmq.Context.instance().destroy()
