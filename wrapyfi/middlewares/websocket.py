import logging
import atexit
import threading

import socketio

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class WebSocketMiddlewarePubSub(metaclass=SingletonOptimized):
    """
    WebSocket middleware wrapper. This class is a singleton, so it can be instantiated only once.
    The `activate` method should be called to initialize the middleware.
    The `deinit` method should be called to deinitialize the middleware and destroy all connections.
    """

    _instance = None  # Singleton instance

    @staticmethod
    def activate(socket_address: str = "http://127.0.0.1:5000", **kwargs):
        """
        Activate the WebSocket middleware. Initializes the WebSocket client with the provided address and options.

        :param socket_address: str: The WebSocket server address
        :param kwargs: dict: Additional keyword arguments for the middleware
        """
        if WebSocketMiddlewarePubSub._instance is None:
            WebSocketMiddlewarePubSub._instance = WebSocketMiddlewarePubSub(
                socket_address=socket_address, **kwargs
            )
        return WebSocketMiddlewarePubSub._instance

    def __init__(
        self,
        socket_address: str = "http://127.0.0.1:5000",
        monitor_listener_spawn: str = None,
        websocket_kwargs: dict = None,
        **kwargs,
    ):
        """
        Initialize the WebSocket middleware. This method is automatically called when the class is instantiated.

        :param socket_address: str: The WebSocket server address
        :param monitor_listener_spawn: str: Determines the type of listener spawn
        :param websocket_kwargs: dict: Additional keyword arguments for the WebSocket connection
        :param kwargs: dict: Additional keyword arguments for compatibility with the interface
        """
        logging.info(f"Initializing WebSocket middleware on {socket_address}")

        # Store arguments, even if unused for now (for interface compatibility)
        self.socket_address = socket_address
        self.monitor_listener_spawn = monitor_listener_spawn
        self.websocket_kwargs = websocket_kwargs or {}

        # Initialize WebSocket client
        self.socketio_client = socketio.Client()

        # Track connection status
        self.connected = False

        # Register event handlers for connection
        @self.socketio_client.event
        def connect():
            logging.info(
                f"[WebSocketMiddlewarePubSub] Connected to {self.socket_address}"
            )
            self.connected = True

        @self.socketio_client.event
        def disconnect():
            logging.info(
                f"[WebSocketMiddlewarePubSub] Disconnected from {self.socket_address}"
            )
            self.connected = False

        # Start the connection in a background thread
        self.client_thread = threading.Thread(target=self._connect_client)
        self.client_thread.daemon = True
        self.client_thread.start()

        # Ensure cleanup at exit
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

    def _connect_client(self):
        """Connect the WebSocket client."""
        try:
            self.socketio_client.connect(
                self.socket_address, namespaces=["/"], **self.websocket_kwargs, retry=True,
            )
            self.socketio_client.wait()  # Wait for messages
        except Exception as e:
            logging.error(
                f"[WebSocketMiddlewarePubSub] Error connecting to {self.socket_address}: {e}"
            )

    def register_callback(self, topic: str, callback):
        """
        Register an event handler for a specific topic.

        :param topic: str: The topic/event to listen to
        :param callback: callable: The function to call when the event occurs
        """
        self.socketio_client.on(topic, callback)
        logging.info(
            f"[WebSocketMiddlewarePubSub] Registered callback for topic {topic}"
        )

    def is_connected(self) -> bool:
        """
        Check whether the WebSocket client is connected.

        :return: bool: True if connected, False otherwise.
        """
        return self.connected

    @staticmethod
    def deinit():
        """
        Deinitialize the WebSocket middleware. This method is automatically called when the program exits.
        """
        logging.info("[WebSocketMiddlewarePubSub] Deinitializing WebSocket client")
        if (
            WebSocketMiddlewarePubSub._instance
            and WebSocketMiddlewarePubSub._instance.socketio_client.connected
        ):
            WebSocketMiddlewarePubSub._instance.socketio_client.disconnect()
            WebSocketMiddlewarePubSub._instance.connected = False
