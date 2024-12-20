import logging
import atexit
import threading
import multiprocessing
import time
from collections import defaultdict
import json
from typing import Optional

import zmq

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator

ZEROMQ_POST_OPTS = [
    "SUBSCRIBE",
    "UNSUBSCRIBE",
    "LINGER",
    "ROUTER_HANDOVER",
    "ROUTER_MANDATORY",
    "PROBE_ROUTER",
    "XPUB_VERBOSE",
    "XPUB_VERBOSER",
    "REQ_CORRELATE",
    "REQ_RELAXED",
    "SNDHWM",
    "RCVHWM",
]


class ZeroMQMiddlewarePubSub(object):
    """
    ZeroMQ PUB/SUB middleware wrapper. This class is a singleton, so it can be instantiated only once. The ``activate``
    method should be called to initialize the middleware. The ``deinit`` method should be called to deinitialize the
    middleware and destroy all connections. The ``activate`` and ``deinit`` methods are automatically called when the
    class is instantiated and when the program exits, respectively.
    """

    class ZeroMQSharedMonitorData:
        """
        Shared data class for the ZeroMQ PUB/SUB monitor. This class is used to share data between the main process and
        the monitor listener process/thread.
        """

        def __init__(self, use_multiprocessing: bool = False):
            """
            Initialize the shared data class.

            :param use_multiprocessing: bool: Whether to use multiprocessing or threading
            """
            self.use_multiprocessing = use_multiprocessing
            if use_multiprocessing:
                self.manager = multiprocessing.Manager()
                self.shared_topics = self.manager.list()
                self.shared_connections = self.manager.dict()
                self.lock = self.manager.Lock()
            else:
                self.shared_topics = []
                self.shared_connections = {}
                self.lock = threading.Lock()

        def add_topic(self, topic: str):
            """
            Add a topic to the shared data class.

            :param topic: str: The topic to add
            """
            with self.lock:
                self.shared_topics.append(topic)

        def remove_topic(self, topic: str):
            """
            Remove a topic from the shared data class.

            :param topic: str: The topic to remove
            """
            try:
                with self.lock:
                    if topic in self.shared_topics:
                        self.shared_topics.remove(topic)
            except (FileNotFoundError, EOFError):
                if self.use_multiprocessing:
                    # TODO(fabawi): this is can break in many ways, and shutting down the topic monitor is not the right solution, since all topics will be affected
                    self.manager.shutdown()

        def get_topics(self):
            """
            Get the list of topics in the shared data class.

            :return: list: The list of topics
            """
            with self.lock:
                return list(self.shared_topics)

        def update_connection(self, topic: str, data: dict):
            """
            Update the connection data for a topic, e.g. the number of subscribers.

            :param topic: str: The topic to update
            :param data: dict: The connection data
            """
            with self.lock:
                self.shared_connections[topic] = data

        def remove_connection(self, topic: str):
            """
            Remove the connection data for a topic.

            :param topic: str: The topic to remove
            """
            with self.lock:
                if topic in list(self.shared_connections.keys()):
                    del self.shared_connections[topic]

        def get_connections(self):
            """
            Get the connection data for all topics.

            :return: dict: The connection data for all topics
            """
            with self.lock:
                return dict(self.shared_connections)

        def is_connected(self, topic: str):
            """
            Check whether a topic is connected.

            :param topic: str: The topic to check
            """
            with self.lock:
                if topic in list(self.shared_connections.keys()):
                    return True
                else:
                    return False

    @staticmethod
    def activate(**kwargs):
        """
        Activate the ZeroMQ PUB/SUB middleware. This method should be called to initialize the middleware.

        :param kwargs: dict: Keyword arguments to be passed to the ZeroMQ initialization function
        """
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

        ZeroMQMiddlewarePubSub(
            zeromq_proxy_kwargs=kwargs,
            zeromq_post_kwargs=zeromq_post_kwargs,
            **zeromq_pre_kwargs,
        )

    def __init__(
        self,
        zeromq_proxy_kwargs: Optional[dict] = None,
        zeromq_post_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the ZeroMQ PUB/SUB middleware. This method is automatically called when the class is instantiated.

        :param zeromq_proxy_kwargs: Optional[dict]: Keyword arguments to be passed to the ZeroMQ proxy initialization function
        :param zeromq_post_kwargs: Optional[dict]: Keyword arguments to be passed to the ZeroMQ initialization function (these are ZeroMQ options)
        :param kwargs: dict: Keyword arguments to be passed to the ZeroMQ initialization function
        """
        self.zeromq_proxy_kwargs = zeromq_proxy_kwargs or {}
        self.zeromq_kwargs = zeromq_post_kwargs or {}
        logging.info("Initialising ZeroMQ PUB/SUB middleware")
        self.ctx = zmq.Context.instance()
        for socket_property in kwargs.items():
            if isinstance(socket_property[1], str):
                self.ctx.setsockopt_string(
                    getattr(zmq, socket_property[0]), socket_property[1]
                )
            else:
                self.ctx.setsockopt(
                    getattr(zmq, socket_property[0]), socket_property[1]
                )
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

        # start the pubsub proxy and monitor
        if zeromq_proxy_kwargs is not None and zeromq_proxy_kwargs:
            # start the pubsub monitor listener
            if zeromq_proxy_kwargs.get("start_pubsub_monitor_broker", False):
                if zeromq_proxy_kwargs["pubsub_monitor_listener_spawn"] == "process":
                    self.shared_monitor_data = self.ZeroMQSharedMonitorData(
                        use_multiprocessing=True
                    )
                    self.monitor = multiprocessing.Process(
                        name="zeromq_pubsub_monitor_listener",
                        target=self.__init_monitor_listener,
                        kwargs=zeromq_proxy_kwargs,
                    )
                    self.monitor.daemon = True
                    self.monitor.start()
                else:  # if threaded
                    self.shared_monitor_data = self.ZeroMQSharedMonitorData(
                        use_multiprocessing=False
                    )
                    self.monitor = threading.Thread(
                        name="pubsub_monitor_listener_spawn",
                        target=self.__init_monitor_listener,
                        kwargs=zeromq_proxy_kwargs,
                    )
                    self.monitor.setDaemon(
                        True
                    )  # deprecation warning Python3.10. Previous Python versions only support this
                    self.monitor.start()

            time.sleep(1)

            if zeromq_proxy_kwargs.get("start_proxy_broker", False):
                if zeromq_proxy_kwargs["proxy_broker_spawn"] == "process":
                    self.proxy = multiprocessing.Process(
                        name="zeromq_pubsub_broker",
                        target=self.__init_proxy,
                        kwargs=zeromq_proxy_kwargs,
                    )
                    self.proxy.daemon = True
                    self.proxy.start()
                else:  # if threaded
                    self.proxy = threading.Thread(
                        name="zeromq_pubsub_broker",
                        target=self.__init_proxy,
                        kwargs=zeromq_proxy_kwargs,
                    )
                    self.proxy.setDaemon(
                        True
                    )  # deprecation warning Python3.10. Previous Python versions only support this
                    self.proxy.start()
            pass

    @staticmethod
    def proxy_thread(
        socket_pub_address: str = "tcp://127.0.0.1:5555",
        socket_sub_address: str = "tcp://127.0.0.1:5556",
        inproc_address: str = "inproc://monitor",
    ):
        """
        Proxy thread for the ZeroMQ PUB/SUB proxy.

        :param socket_pub_address: str: The address of the PUB socket
        :param socket_sub_address: str: The address of the SUB socket
        :param inproc_address: str: The address of the inproc socket (connections within the same process, for exchanging subscription data between the proxy and the monitor)
        """
        context = zmq.Context.instance()
        xpub = context.socket(zmq.XPUB)
        xsub = context.socket(zmq.XSUB)
        xpub.setsockopt(zmq.XPUB_VERBOSE, 1)

        xpub.bind(socket_pub_address)
        xsub.bind(socket_sub_address)

        monitor = context.socket(zmq.PUB)
        monitor.bind(inproc_address)
        try:
            zmq.proxy(xpub, xsub, monitor)
        except Exception as e:
            # WORKAROUND(fabawi): The errno in ZMQERROR changes arbitrarily but the message seems consistent. According
            #  to https://pyzmq.readthedocs.io/en/latest/api/zmq.html#context when ctx.destroy() is called, and the
            #  context is created in another thread, the operation is unsafe. If we want to maintain daemonized spawning
            #  of brokers, we have to apply this workaround. There is no way to let the proxy know that the context has
            #  been destroyed, so we let it crash. This only happens when the spawning method is `thread`; `process`
            #  (default method) will hang forever. This can be done by setting the environment variables
            #  WRAPYFI_ZEROMQ_PROXY_BROKER_SPAWN=thread WRAPYFI_ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN=thread
            if str(e) == "Socket operation on non-socket":
                pass
            else:
                logging.error(
                    f"[ZeroMQ BROKER] An error occurred in the ZeroMQ proxy: {str(e)}."
                )

    @staticmethod
    def subscription_monitor_thread(
        inproc_address: str = "inproc://monitor",
        socket_sub_address: str = "tcp://127.0.0.1:5556",
        pubsub_monitor_topic: str = "ZEROMQ/CONNECTIONS",
        verbose: bool = False,
    ):
        """
        Subscription monitor thread for the ZeroMQ PUB/SUB proxy.

        :param inproc_address: str: The address of the inproc socket (connections within the same process, for exchanging subscription data between the proxy and the monitor)
        :param socket_sub_address: str: The address of the SUB socket
        :param pubsub_monitor_topic: str: The topic to use for publishing subscription data
        :param verbose: bool: Whether to print debug messages
        """
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
                    topic = message[1:].decode("utf-8")

                    if verbose:
                        logging.info(
                            f"[ZeroMQ BROKER] Received event: {event}, topic: {topic}"
                        )

                    # avoid processing messages on the monitor topic.
                    if topic == pubsub_monitor_topic:
                        continue

                    # update the count of subscribers for the topic
                    if event == 1:  # subscribe
                        topic_subscriber_count[topic] += 1
                    elif event == 0:  # unsubscribe
                        topic_subscriber_count[topic] = 0

                    if verbose:
                        logging.info(
                            f"[ZeroMQ BROKER] Current topic subscriber count: {dict(topic_subscriber_count)}"
                        )

                    # publish the updated counts
                    publisher.send_multipart(
                        [
                            pubsub_monitor_topic.encode(),
                            json.dumps(dict(topic_subscriber_count)).encode(),
                        ]
                    )
            except Exception as e:
                logging.error(
                    f"[ZeroMQ BROKER] An error occurred in the ZeroMQ subscription monitor: {str(e)}"
                )

    def __init_proxy(
        self,
        socket_pub_address: str = "tcp://127.0.0.1:5555",
        socket_sub_address: str = "tcp://127.0.0.1:5556",
        pubsub_monitor_topic: str = "ZEROMQ/CONNECTIONS",
        **kwargs,
    ):
        """
        Initialize the ZeroMQ PUB/SUB proxy and subscription monitor.

        :param socket_pub_address: str: The address of the PUB socket
        :param socket_sub_address: str: The address of the SUB socket
        :param pubsub_monitor_topic: str: The topic to use for publishing subscription data
        :param kwargs: dict: Keyword arguments to be passed to the ZeroMQ initialization function
        """
        inproc_address = "inproc://monitor"

        threading.Thread(
            target=self.proxy_thread,
            kwargs={
                "socket_pub_address": socket_pub_address,
                "socket_sub_address": socket_sub_address,
                "inproc_address": inproc_address,
            },
        ).start(),

        threading.Thread(
            target=self.subscription_monitor_thread,
            kwargs={
                "socket_sub_address": socket_sub_address,
                "inproc_address": inproc_address,
                "pubsub_monitor_topic": pubsub_monitor_topic,
                "verbose": kwargs.get("verbose", False),
            },
        ).start()

    def __init_monitor_listener(
        self,
        socket_pub_address: str = "tcp://127.0.0.1:5555",
        pubsub_monitor_topic: str = "ZEROMQ/CONNECTIONS",
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the ZeroMQ PUB/SUB monitor listener.

        :param socket_pub_address: str: The address of the PUB socket
        :param pubsub_monitor_topic: str: The topic to use for publishing subscription data
        :param verbose: bool: Whether to print debug messages
        """
        try:
            context = zmq.Context()
            subscriber = context.socket(zmq.SUB)

            subscriber.connect(socket_pub_address)
            subscriber.setsockopt_string(zmq.SUBSCRIBE, pubsub_monitor_topic)

            while True:
                _, message = subscriber.recv_multipart()
                data = json.loads(message.decode("utf-8"))

                for topic, value in data.items():
                    if verbose:
                        logging.info(f"[ZeroMQ] Topic: {topic}, Data: {value}")

                    # check if the topic exists in shared monitor data
                    if topic in self.shared_monitor_data.get_topics():
                        self.shared_monitor_data.update_connection(topic, value)
                        if value == 0:
                            logging.info(
                                f"[ZeroMQ] Subscriber disconnected from topic: {topic}"
                            )
                            self.shared_monitor_data.remove_connection(topic)
                        else:
                            logging.info(
                                f"[ZeroMQ] Subscriber connected to topic: {topic}"
                            )

                if verbose:
                    for monitored_topic in self.shared_monitor_data.get_topics():
                        logging.info(
                            f"[ZeroMQ] Monitored topic from main process: {monitored_topic}"
                        )

        except Exception as e:
            logging.error(
                f"[ZeroMQ] An error occurred in the ZeroMQ subscription monitor listener: {str(e)}"
            )

    @staticmethod
    def deinit():
        logging.info("Deinitializing ZeroMQ middleware")
        zmq.Context.instance().destroy()


class ZeroMQMiddlewarePubSubListen(
    ZeroMQMiddlewarePubSub, metaclass=SingletonOptimized
):

    @staticmethod
    def activate(**kwargs):
        """
        Activate the ZeroMQ PUB/SUB middleware. This method should be called to initialize the middleware.

        :param kwargs: dict: Keyword arguments to be passed to the ZeroMQ initialization function
        """
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

        ZeroMQMiddlewarePubSubListen(
            zeromq_proxy_kwargs=kwargs,
            zeromq_post_kwargs=zeromq_post_kwargs,
            **zeromq_pre_kwargs,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ZeroMQMiddlewarePubSubPublish(
    ZeroMQMiddlewarePubSub, metaclass=SingletonOptimized
):
    @staticmethod
    def activate(**kwargs):
        """
        Activate the ZeroMQ PUB/SUB middleware. This method should be called to initialize the middleware.

        :param kwargs: dict: Keyword arguments to be passed to the ZeroMQ initialization function
        """
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

        ZeroMQMiddlewarePubSubPublish(
            zeromq_proxy_kwargs=kwargs,
            zeromq_post_kwargs=zeromq_post_kwargs,
            **zeromq_pre_kwargs,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ZeroMQMiddlewareReqRep(metaclass=SingletonOptimized):
    """
    ZeroMQ REQ/REP middleware wrapper. This class is a singleton, so it can be instantiated only once. The ``activate``
    method should be called to initialize the middleware. The ``deinit`` method should be called to deinitialize the
    middleware and destroy all connections. The ``activate`` and ``deinit`` methods are automatically called when the
    class is instantiated and when the program exits, respectively.
    """

    @staticmethod
    def activate(**kwargs):
        """
        Activate the ZeroMQ REQ/REP middleware. This method should be called to initialize the middleware.

        :param kwargs: dict: Keyword arguments to be passed to the ZeroMQ initialization function
        """
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

        ZeroMQMiddlewareReqRep(
            zeromq_proxy_kwargs=kwargs,
            zeromq_post_kwargs=zeromq_post_kwargs,
            **zeromq_pre_kwargs,
        )

    def __init__(
        self,
        zeromq_proxy_kwargs: Optional[dict] = None,
        zeromq_post_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the ZeroMQ REQ/REP middleware. This method is automatically called when the class is instantiated.

        :param zeromq_proxy_kwargs: Optional[dict]: Keyword arguments to be passed to the ZeroMQ proxy initialization function
        :param zeromq_post_kwargs: Optional[dict]: Keyword arguments to be passed to the ZeroMQ initialization function (these are ZeroMQ options)
        :param args: list: Positional arguments to be passed to the ZeroMQ initialization function
        :param kwargs: dict: Keyword arguments to be passed to the ZeroMQ initialization function
        """
        self.zeromq_proxy_kwargs = zeromq_proxy_kwargs or {}
        self.zeromq_kwargs = zeromq_post_kwargs or {}
        logging.info("Initialising ZeroMQ REQ/REP middleware")
        self.ctx = zmq.Context.instance()
        for socket_property in kwargs.items():
            if isinstance(socket_property[1], str):
                self.ctx.setsockopt_string(
                    getattr(zmq, socket_property[0]), socket_property[1]
                )
            else:
                self.ctx.setsockopt(
                    getattr(zmq, socket_property[0]), socket_property[1]
                )
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

        if zeromq_proxy_kwargs is not None and zeromq_proxy_kwargs:
            if zeromq_proxy_kwargs["proxy_broker_spawn"] == "process":
                self.proxy = multiprocessing.Process(
                    name="zeromq_reqrep_broker",
                    target=self.__init_device,
                    kwargs=zeromq_proxy_kwargs,
                )
                self.proxy.daemon = True
                self.proxy.start()
            else:  # if threaded
                self.proxy = threading.Thread(
                    name="zeromq_reqrep_broker",
                    target=self.__init_device,
                    kwargs=zeromq_proxy_kwargs,
                )
                self.proxy.setDaemon(
                    True
                )  # deprecation warning Python3.10. Previous Python versions only support this
                self.proxy.start()
            pass

    @staticmethod
    def __init_device(
        socket_rep_address: str = "tcp://127.0.0.1:5559",
        socket_req_address: str = "tcp://127.0.0.1:5560",
        **kwargs,
    ):
        """
        Initialize the ZeroMQ REQ/REP device broker.

        :param socket_rep_address: str: The address of the REP socket
        :param socket_req_address: str: The address of the REQ socket
        """
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
        # logging.info(f"[ZeroMQ] Intialising REQ/REP device broker")
        try:
            zmq.proxy(xrep, xreq)
        except Exception as e:
            # WORKAROUND(fabawi): The errno in ZMQERROR changes arbitrarily but the message seems consistent. According
            #  to https://pyzmq.readthedocs.io/en/latest/api/zmq.html#context when ctx.destroy() is called, and the
            #  context is created in another thread, the operation is unsafe. If we want to maintain daemonized spawning
            #  of brokers, we have to apply this workaround. There is no way to let the proxy know that the context has
            #  been destroyed, so we let it crash. This only happens when the spawning method is `thread`; `process`
            #  (default method) will hang forever. This can be done by setting the environment variables
            #  WRAPYFI_ZEROMQ_PROXY_BROKER_SPAWN=thread WRAPYFI_ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN=thread
            if str(e) == "Socket operation on non-socket":
                pass
            else:
                logging.error(
                    f"[ZeroMQ] An error occurred in the ZeroMQ proxy: {str(e)}."
                )

    @staticmethod
    def deinit():
        logging.info("Deinitializing ZeroMQ middleware")
        zmq.Context.instance().destroy()


class ZeroMQMiddlewareParamServer(metaclass=SingletonOptimized):
    """
    ZeroMQ parameter server middleware wrapper. This class is a singleton, so it can be instantiated only once. The
    ``activate`` method should be called to initialize the middleware. The ``deinit`` method should be called to
    deinitialize the middleware and destroy all connections. The ``activate`` and ``deinit`` methods are automatically
    called when the class is instantiated and when the program exits, respectively.

    Note: This parameter server is experimental and not fully tested.
    """

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

        ZeroMQMiddlewareParamServer(
            zeromq_proxy_kwargs=kwargs,
            zeromq_post_kwargs=zeromq_post_kwargs,
            **zeromq_pre_kwargs,
        )

    def __init__(
        self,
        zeromq_proxy_kwargs: Optional[dict] = None,
        zeromq_post_kwargs: Optional = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the ZeroMQ parameter server middleware. This method is automatically called when the class is
        instantiated.

        :param zeromq_proxy_kwargs: Optional[dict]: Keyword arguments to be passed to the ZeroMQ proxy initialization function
        :param zeromq_post_kwargs: Optional[dict]: Keyword arguments to be passed to the ZeroMQ initialization function (these are ZeroMQ options)
        :param kwargs: dict: Keyword arguments to be passed to the ZeroMQ initialization function
        """
        self.zeromq_proxy_kwargs = zeromq_proxy_kwargs or {}
        self.zeromq_kwargs = zeromq_post_kwargs or {}
        logging.info("Initialising ZeroMQ Parameter Server")
        self.ctx = zmq.Context.instance()
        for socket_property in kwargs.items():
            if isinstance(socket_property[1], str):
                self.ctx.setsockopt_string(
                    getattr(zmq, socket_property[0]), socket_property[1]
                )
            else:
                self.ctx.setsockopt(
                    getattr(zmq, socket_property[0]), socket_property[1]
                )

        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

        if zeromq_proxy_kwargs is not None and zeromq_proxy_kwargs:
            self.manager = multiprocessing.Manager()
            self.params = self.manager.dict()
            self.params["WRAPYFI_ACTIVE"] = "True"
            if zeromq_proxy_kwargs["proxy_broker_spawn"] == "process":
                self.param_broadcaster = multiprocessing.Process(
                    name="zeromq_param_broadcaster",
                    target=self.__init_broadcaster,
                    kwargs=zeromq_proxy_kwargs,
                    args=(self.params,),
                )
                self.param_broadcaster.daemon = True
                self.param_broadcaster.start()
                self.param_server = multiprocessing.Process(
                    name="zeromq_param_server",
                    target=self.__init_server,
                    kwargs=zeromq_proxy_kwargs,
                    args=(self.params,),
                )
                self.param_server.daemon = True
                self.param_server.start()
            else:  # if threaded

                self.param_broadcaster = threading.Thread(
                    name="zeromq_param_broadcaster",
                    target=self.__init_broadcaster,
                    kwargs=zeromq_proxy_kwargs,
                    args=(self.params,),
                )
                self.param_broadcaster.setDaemon(
                    True
                )  # deprecation warning Python3.10. Previous Python versions only support this
                self.param_broadcaster.start()
                self.param_server = threading.Thread(
                    name="zeromq_param_server",
                    target=self.__init_server,
                    kwargs=zeromq_proxy_kwargs,
                    args=(self.params,),
                )
                self.param_server.setDaemon(
                    True
                )  # deprecation warning Python3.10. Previous Python versions only support this
                self.param_server.start()
            pass

    @staticmethod
    def __init_broadcaster(
        params,
        param_pub_address: str = "tcp://127.0.0.1:5655",
        param_sub_address: str = "tcp://127.0.0.1:5656",
        param_poll_interval=1,
        verbose=False,
        **kwargs,
    ):
        """
        Initialize the ZeroMQ parameter server broadcaster.

        :param params: dict: The parameters to be broadcasted
        :param param_pub_address: str: The address of the PUB socket
        :param param_sub_address: str: The address of the SUB socket
        :param param_poll_interval: int: The polling interval for the parameter server
        :param verbose: bool: Whether to print debug messages
        """
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
                    logging.info(
                        "[ZeroMQ BROKER] xpub_socket recv message: %r" % message
                    )
                if message[0].startswith(b"\x00"):
                    root_topics.remove(message[0][1:].decode("utf-8"))
                elif message[0].startswith(b"\x01"):
                    root_topics.add(message[0][1:].decode("utf-8"))
                xsub_socket.send_multipart(message)

            if xsub_socket in event:
                message = xsub_socket.recv_multipart()
                if verbose:
                    logging.info(
                        "[ZeroMQ BROKER] xsub_socket recv message: %r" % message
                    )
                if message[0].startswith(b"\x01") or message[0].startswith(b"\x00"):
                    xpub_socket.send_multipart(message)
                else:
                    fltr_key = message[0].decode("utf-8")
                    fltr_message = {
                        key: val
                        for key, val in params.items()
                        if key.startswith(fltr_key)
                    }
                    if verbose:
                        logging.info(
                            "[ZeroMQ BROKER] xsub_socket filtered message: %r"
                            % fltr_message
                        )
                    for key, val in fltr_message.items():
                        prefix, param = key.rsplit("/", 1) if "/" in key else ("", key)
                        xpub_socket.send_multipart(
                            [
                                prefix.encode("utf-8"),
                                param.encode("utf-8"),
                                val.encode("utf-8"),
                            ]
                        )
                # xpub_socket.send_multipart(message)

            if event:
                update_trigger = True

            if param_server is not None:
                update_trigger, cached_params = (
                    ZeroMQMiddlewareParamServer.publish_params(
                        param_server, params, cached_params, root_topics, update_trigger
                    )
                )

    @staticmethod
    def publish_params(
        param_server,
        params: dict,
        cached_params: dict,
        root_topics: set,
        update_trigger: bool,
    ):
        """
        Publish parameters to the parameter server.

        :param param_server: zmq.Socket: The parameter server socket
        :param params: dict: The parameters to be published
        :param cached_params: dict: The cached parameters. This is used to check whether the parameters have changed
        :param root_topics: set: The root topics. This is used to check whether there are any active subscribers
        :param update_trigger: bool: Whether to trigger an update of the parameters
        :return: Tuple[bool, dict]: Whether to trigger an update of the parameters and the cached parameters
        """
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
            param_server.send_multipart(
                [prefix.encode("utf-8"), param.encode("utf-8"), val.encode("utf-8")]
            )

        return update_trigger, cached_params

    @staticmethod
    def __init_server(
        params: dict, param_reqrep_address: str = "tcp://127.0.0.1:5659", **kwargs
    ):
        """
        Initialize the ZeroMQ parameter server.

        :param params: dict: The parameters to be published
        :param param_reqrep_address: str: The address of the REQ/REP socket
        """
        ctx = zmq.Context.instance()
        request_server = ctx.socket(zmq.REP)
        request_server.bind(param_reqrep_address)

        while True:
            request = request_server.recv_string()
            if request.startswith("get"):
                try:
                    # extract the parameter name and namespace prefix from the request
                    prefix, param = (
                        request[4:].rsplit("/", 1)
                        if "/" in request[4:]
                        else ("", request[4:])
                    )
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
        """
        Deinitialize the ZeroMQ parameter server.
        """
        logging.info("Deinitializing ZeroMQ Parameter Server")
        zmq.Context.instance().destroy()
