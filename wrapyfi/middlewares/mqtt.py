import logging
import atexit
import threading

import paho.mqtt.client as mqtt_client

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class MqttMiddlewarePubSub(metaclass=SingletonOptimized):
    """
    MQTT middleware wrapper. This class is a singleton, so it can be instantiated only once.
    The `activate` method should be called to initialize the middleware.
    The `deinit` method should be called to deinitialize the middleware and destroy all connections.
    """

    _instance = None  # Singleton instance

    @staticmethod
    def activate(
        broker_address: str = "broker.emqx.io",
        broker_port: int = 1883,
        **kwargs,
    ):
        """
        Activate the MQTT middleware. Initializes the MQTT client with the provided address and options.

        :param broker_address: str: The MQTT broker address
        :param broker_port: int: The MQTT broker port
        :param kwargs: dict: Additional keyword arguments for the middleware
        """
        if MqttMiddlewarePubSub._instance is None:
            MqttMiddlewarePubSub._instance = MqttMiddlewarePubSub(
                broker_address=broker_address, broker_port=broker_port, **kwargs
            )
        return MqttMiddlewarePubSub._instance

    def __init__(
        self,
        broker_address: str = "broker.emqx.io",
        broker_port: int = 1883,
        client_id: str = None,
        **kwargs,
    ):
        """
        Initialize the MQTT middleware. This method is automatically called when the class is instantiated.

        :param broker_address: str: The MQTT broker address
        :param broker_port: int: The MQTT broker port
        :param client_id: str: The MQTT client ID
        :param kwargs: dict: Additional keyword arguments for compatibility with the interface
        """
        logging.info(f"Initializing MQTT middleware on {broker_address}:{broker_port}")

        # Store arguments, even if unused (for compatibility)
        self.broker_address = broker_address
        self.port = broker_port
        self.client_id = client_id

        # Create a MQTT client
        self.mqtt_client = mqtt_client.Client(
            mqtt_client.CallbackAPIVersion.VERSION2, client_id=self.client_id
        )

        # Set up callbacks
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_disconnect = self._on_disconnect
        self.mqtt_client.on_message = self._on_message

        # Dictionary to store topic-specific callbacks
        self.topic_callbacks = {}

        # Track connection status
        self.connected = False

        # Start the connection in a background thread
        self.client_thread = threading.Thread(target=self._connect_client)
        self.client_thread.daemon = True
        self.client_thread.start()

        # Ensure cleanup at exit
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

    def _connect_client(self):
        """Connect the MQTT client."""
        try:
            self.mqtt_client.connect(self.broker_address, self.port)
            # Use loop_start instead of loop_forever for non-blocking behavior
            self.mqtt_client.loop_start()
        except Exception as e:
            logging.error(
                f"[MqttMiddlewarePubSub] Error connecting to {self.broker_address}:{self.port}: {e}"
            )

    def _on_connect(self, client, userdata, flags, rc, properties):
        """Callback for when the MQTT client connects to the broker."""
        logging.info(
            f"[MqttMiddlewarePubSub] Connected to {self.broker_address}:{self.port}"
        )
        self.connected = True

    def _on_disconnect(self, client, userdata, rc):
        """Callback for when the MQTT client disconnects from the broker."""
        logging.info(
            f"[MqttMiddlewarePubSub] Disconnected from {self.broker_address}:{self.port}"
        )
        self.connected = False

    def _on_message(self, client, userdata, msg):
        """Callback for when the MQTT client receives a message."""
        if msg.topic in self.topic_callbacks:
            callback = self.topic_callbacks[msg.topic]
            callback(client, userdata, msg)

    def register_callback(self, topic: str, callback):
        """
        Register an event handler for a specific topic.

        :param topic: str: The topic to subscribe to
        :param callback: callable: The function to call when a message is received on this topic
        """
        self.mqtt_client.subscribe(topic)
        self.topic_callbacks[topic] = callback
        logging.info(f"[MqttMiddlewarePubSub] Registered callback for topic {topic}")

    def is_connected(self) -> bool:
        """
        Check whether the MQTT client is connected.

        :return: bool: True if connected, False otherwise.
        """
        return self.connected

    @staticmethod
    def deinit():
        """
        Deinitialize the MQTT middleware. This method is automatically called when the program exits.
        """
        logging.info("[MqttMiddlewarePubSub] Deinitializing MQTT client")
        if MqttMiddlewarePubSub._instance:
            MqttMiddlewarePubSub._instance.mqtt_client.loop_stop()
            MqttMiddlewarePubSub._instance.mqtt_client.disconnect()
            MqttMiddlewarePubSub._instance.connected = False
