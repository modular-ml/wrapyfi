import logging
import atexit
import threading
import multiprocessing
import time
from collections import defaultdict
import json

import zmq

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator

ZEROMQ_POST_OPTS = ["SUBSCRIBE", "UNSUBSCRIBE", "LINGER", "ROUTER_HANDOVER", "ROUTER_MANDATORY", "PROBE_ROUTER",
                    "XPUB_VERBOSE", "XPUB_VERBOSER", "REQ_CORRELATE", "REQ_RELAXED", "SNDHWM", "RCVHWM"]


class ZeroMQMiddlewarePubSub(metaclass=SingletonOptimized):
    class ZeroMQSharedMonitorData:
        def __init__(self, use_multiprocessing=False):
            self.use_multiprocessing = use_multiprocessing
            if use_multiprocessing:
                manager = multiprocessing.Manager()
                self.shared_topics = manager.list()
                self.shared_connections = manager.dict()
                self.lock = manager.Lock()
            else:
                self.shared_topics = []
                self.shared_connections = {}
                self.lock = threading.Lock()

        def add_topic(self, topic):
            with self.lock:
                self.shared_topics.append(topic)

        def remove_topic(self, topic):
            with self.lock:
                if topic in self.shared_topics:
                    self.shared_topics.remove(topic)

        def get_topics(self):
            with self.lock:
                return list(self.shared_topics)

        def update_connection(self, topic, data):
            with self.lock:
                self.shared_connections[topic] = data

        def get_connections(self):
            with self.lock:
                return dict(self.shared_connections)

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

        # start the pubsub proxy and monitor
        if zeromq_proxy_kwargs is not None and zeromq_proxy_kwargs:
            if zeromq_proxy_kwargs.get("start_proxy_broker", False):
                if zeromq_proxy_kwargs["proxy_broker_spawn"] == "process":
                    self.proxy = multiprocessing.Process(name='zeromq_pubsub_broker', target=self.__init_proxy, kwargs=zeromq_proxy_kwargs)
                    self.proxy.daemon = True
                    self.proxy.start()
                else:  # if threaded
                    self.proxy = threading.Thread(name='zeromq_pubsub_broker', target=self.__init_proxy, kwargs=zeromq_proxy_kwargs)
                    self.proxy.setDaemon(True)
                    self.proxy.start()
            pass
        
            # start the pubsub monitor listener
            if zeromq_proxy_kwargs.get("pubsub_monitor_listener_spawn", False):
                if zeromq_proxy_kwargs["pubsub_monitor_listener_spawn"] == "process":
                    self.shared_monitor_data = self.ZeroMQSharedMonitorData(use_multiprocessing=True)
                    self.monitor = multiprocessing.Process(name='zeromq_pubsub_monitor_listener', target=self.__init_monitor_listener, kwargs=zeromq_proxy_kwargs)
                    self.monitor.daemon = True
                    self.monitor.start()
                else:  # if threaded
                    self.shared_monitor_data = self.ZeroMQSharedMonitorData(use_multiprocessing=False)
                    self.monitor = threading.Thread(name='pubsub_monitor_listener_spawn', target=self.__init_monitor_listener, kwargs=zeromq_proxy_kwargs)
                    self.monitor.setDaemon(True)
                    self.monitor.start()
                
    @staticmethod
    def proxy_thread(socket_pub_address="tcp://127.0.0.1:5555",
                     socket_sub_address="tcp://127.0.0.1:5556",
                     inproc_address="inproc://monitor"):
        context = zmq.Context.instance()
        xpub = context.socket(zmq.XPUB)
        xsub = context.socket(zmq.XSUB)
        xpub.setsockopt(zmq.XPUB_VERBOSE, 1)

        xpub.bind(socket_pub_address)
        xsub.bind(socket_sub_address)

        monitor = context.socket(zmq.PUB)
        monitor.bind(inproc_address)

        zmq.proxy(xpub, xsub, monitor)

    @staticmethod
    def subscription_monitor_thread(inproc_address="inproc://monitor", socket_sub_address="tcp://127.0.0.1:5556",
                                    pubsub_monitor_topic="ZEROMQ/CONNECTIONS", verbose=False):
        context = zmq.Context.instance()
        subscriber = context.socket(zmq.SUB)
        subscriber.connect(inproc_address)
        subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

        # pub socket to publish subscriber counts.
        publisher = context.socket(zmq.PUB)
        publisher.connect(socket_sub_address)

        topic_subscriber_count = defaultdict(int)

        while True:
            time.sleep(1)
            try:
                message = subscriber.recv()
                if verbose:
                    logging.info(f"[ZeroMQ BROKER] Raw message: {message}")

                # ensure the message is a subscription/unsubscription message
                if len(message) > 1 and (message[0] == 1 or message[0] == 0):
                    event = message[0]
                    topic = message[1:].decode('utf-8')

                    if verbose:
                        logging.info(f"[ZeroMQ BROKER] Received event: {event}, topic: {topic}")

                    # avoid processing messages on the monitor topic.
                    if topic == pubsub_monitor_topic:
                        continue

                    # update the count of subscribers for the topic
                    if event == 1:  # subscribe
                        topic_subscriber_count[topic] += 1
                    elif event == 0:  # unsubscribe
                        topic_subscriber_count[topic] -= 1

                    if verbose:
                        logging.info(f"[ZeroMQ BROKER] Current topic subscriber count: {dict(topic_subscriber_count)}")

                    # publish the updated counts
                    publisher.send_multipart(
                        [pubsub_monitor_topic.encode(), json.dumps(dict(topic_subscriber_count)).encode()]
                    )
            except Exception as e:
                logging.error(f"[ZeroMQ BROKER] An error occurred in the ZeroMQ subscription monitor: {str(e)}")

    def __init_proxy(self, socket_pub_address="tcp://127.0.0.1:5555", socket_sub_address="tcp://127.0.0.1:5556",
                     pubsub_monitor_topic="ZEROMQ/CONNECTIONS",
                     **kwargs):
        inproc_address = "inproc://monitor"

        threading.Thread(target=self.proxy_thread,
                         kwargs={"socket_pub_address": socket_pub_address,
                                 "socket_sub_address": socket_sub_address,
                                 "inproc_address": inproc_address}).start(),

        threading.Thread(target=self.subscription_monitor_thread,
                         kwargs={"socket_sub_address": socket_sub_address,
                                 "inproc_address": inproc_address,
                                 "pubsub_monitor_topic": pubsub_monitor_topic,
                                 "verbose": kwargs.get("verbose", False)}).start()

    def __init_monitor_listener(self, socket_pub_address="tcp://127.0.0.1:5555", 
                                pubsub_monitor_topic="ZEROMQ/CONNECTIONS",
                                verbose=False, **kwargs):
        try:
            context = zmq.Context()
            subscriber = context.socket(zmq.SUB)

            subscriber.connect(socket_pub_address)
            subscriber.setsockopt_string(zmq.SUBSCRIBE, pubsub_monitor_topic)

            while True:
                _, message = subscriber.recv_multipart()
                data = json.loads(message.decode('utf-8'))
                topic = list(data.keys())[0]
                if verbose:
                    logging.info(f"[ZeroMQ] Topic: {topic}, Data: {data}")

                if topic in self.shared_monitor_data.get_topics():
                    self.shared_monitor_data.update_connection(topic, data)
                    logging.info(f"[ZeroMQ] {data[topic]} subscribers connected to topic: {topic}")

                if verbose:
                    for monitored_topic in self.shared_monitor_data.get_topics():
                        logging.info(f"[ZeroMQ] Monitored topic from main process: {monitored_topic}")

        except Exception as e:
            logging.error(f"[ZeroMQ] An error occurred in the ZeroMQ subscription monitor listener: {str(e)}")

    @staticmethod
    def deinit():
        logging.info("Deinitialising ZeroMQ middleware")
        zmq.Context.instance().destroy()


class ZeroMQMiddlewareReqRep(metaclass=SingletonOptimized):

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

        ZeroMQMiddlewareReqRep(zeromq_proxy_kwargs=kwargs, zeromq_post_kwargs=zeromq_post_kwargs, **zeromq_pre_kwargs)

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
                self.proxy = multiprocessing.Process(name='zeromq_reqrep_broker', target=self.__init_device, kwargs=zeromq_proxy_kwargs)
                self.proxy.daemon = True
                self.proxy.start()
            else:  # if threaded
                self.proxy = threading.Thread(name='zeromq_reqrep_broker', target=self.__init_device, kwargs=zeromq_proxy_kwargs)
                self.proxy.setDaemon(True)
                self.proxy.start()
            pass

    @staticmethod
    def __init_device(socket_rep_address="tcp://127.0.0.1:5559", socket_req_address="tcp://127.0.0.1:5560", **kwargs):
        xrep = zmq.Context.instance().socket(zmq.XREP)
        try:
            xrep.bind(socket_rep_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_rep_address}")
            return
        xreq = zmq.Context.instance().socket(zmq.XREQ)
        try:
            xreq.bind(socket_req_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_req_address}")
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
        # check if there are any subscribed clients before publishing updates
        if not root_topics:
            return None, None

        if not any((update_trigger, root_topics)) and cached_params == params:
            return None, None
        else:
            time.sleep(0.01)
            update_trigger = False
            cached_params = params.copy()

        # publish updates for all parameters to subscribed clients
        for key, val in params.items():
            prefix, param = key.rsplit("/", 1) if "/" in key else ("", key)
            param_server.send_multipart([prefix.encode("utf-8"), param.encode("utf-8"), val.encode("utf-8")])

        return update_trigger, cached_params

    @staticmethod
    def __init_server(params, param_reqrep_address="tcp://127.0.0.1:5659", **kwargs):
        ctx = zmq.Context.instance()
        request_server = ctx.socket(zmq.REP)
        request_server.bind(param_reqrep_address)

        while True:
            request = request_server.recv_string()
            if request.startswith("get"):
                try:
                    # extract the parameter name and namespace prefix from the request
                    prefix, param = request[4:].rsplit("/", 1) if "/" in request[4:] else ("", request[4:])
                    # construct the full parameter name with the namespace prefix
                    full_param = "/".join([prefix, param]) if prefix else param
                    if full_param in params:
                        request_server.send_string(str(params[full_param]))
                    else:
                        request_server.send_string("error:::parameter does not exist")
                except ValueError:
                    request_server.send_string("error:::malformed request")
            elif request.startswith("read"):
                try:
                    # extract the parameter name and namespace prefix from the request
                    prefix = request[5:]
                    # construct the full parameter name with the namespace prefix
                    if any(param.startswith(prefix) for param in params.keys()):
                        request_server.send_string(f"success:::{prefix}")
                    else:
                        request_server.send_string("error:::parameter does not exist")
                except ValueError:
                    request_server.send_string("error:::malformed request")
            elif request.startswith("set"):
                try:
                    # extract the parameter name, namespace prefix and value from the request
                    prefix, param, value = request[4:].rsplit("/", 2)
                    # construct the full parameter name with the namespace prefix
                    full_param = "/".join([prefix, param]) if prefix else param
                    params[full_param] = value
                    request_server.send_string(f"success:::{prefix}")
                except ValueError:
                    request_server.send_string("error:::malformed request")
            elif request.startswith("delete"):
                try:
                    # extract the parameter name and namespace prefix from the request
                    prefix, param = request[7:].rsplit("/", 1)
                    # construct the full parameter name with the namespace prefix
                    full_param = "/".join([prefix, param]) if prefix else param
                    if full_param in params:
                        del params[full_param]
                        request_server.send_string(f"success:::{prefix}")
                    else:
                        request_server.send_string("error:::parameter does not exist")
                except ValueError:
                    request_server.send_string("error:::malformed request")
            else:
                request_server.send_string("error:::invalid request")

    @staticmethod
    def deinit():
        logging.info("Deinitialising ZeroMQ Parameter Server")
        zmq.Context.instance().destroy()
