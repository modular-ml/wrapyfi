import logging
import threading
import atexit
import json

import zenoh

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class ZenohMiddlewarePubSub(metaclass=SingletonOptimized):
    """
    Zenoh middleware wrapper with singleton pattern.
    The `activate` method initializes the middleware, and `deinit` handles cleanup.

    Configurations are initialized by merging any provided keyword arguments,
    environment variables, and direct `zenoh.Config` parameters.
    """

    _instance = None  # Singleton instance

    @staticmethod
    def activate(config: zenoh.Config = None, **kwargs):
        """
        Activates the Zenoh middleware. Initializes the Zenoh session by merging a provided configuration,
        environment variables, and additional keyword arguments.

        :param config: zenoh.Config: Optional Zenoh configuration; merged with environment and `kwargs`
        :param kwargs: dict: Additional settings for customization
        :return: ZenohMiddlewarePubSub instance
        """
        zenoh.init_log_from_env_or("error")
        if ZenohMiddlewarePubSub._instance is None:
            ZenohMiddlewarePubSub._instance = ZenohMiddlewarePubSub(config=config, **kwargs)
        return ZenohMiddlewarePubSub._instance

    def __init__(self, config: zenoh.Config = None, **kwargs):
        """
        Initializes the Zenoh session and sets up a clean exit with deinitialization.

        :param config: zenoh.Config: Configuration for Zenoh session
        """
        logging.info("Initializing Zenoh middleware")

        # Initialize Zenoh session with configuration or default values
        self.config = config or self._merge_config_with_env(kwargs)
        self.session = zenoh.open(self.config)
        self.subscribers = {}

        # Ensure cleanup at exit
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

    def _merge_config_with_env(self, config_kwargs):
        """
        Merges given configuration parameters with environment-based defaults.

        :param config_kwargs: dict: Direct configuration values to merge
        :return: zenoh.Config: Complete Zenoh configuration instance
        """
        config = zenoh.Config()
        for key, value in config_kwargs.items():
            config.insert_json5(key, json.dumps(value))
        return config

    def register_callback(self, topic: str, callback):
        """
        Registers an event handler for a specific topic.

        :param topic: str: The topic to subscribe to
        :param callback: callable: Function to call upon receiving a message
        """
        if topic not in self.subscribers:
            self.subscribers[topic] = self.session.declare_subscriber(topic, callback)
        logging.info(f"[ZenohMiddlewarePubSub] Registered callback for topic {topic}")

    def is_connected(self) -> bool:
        """
        Checks if the Zenoh session is active.

        :return: bool: True if connected, False otherwise
        """
        return self.session is not None and not self.session.is_closed()

    def deinit(self):
        """
        Closes the Zenoh session upon exit.
        """
        logging.info("[ZenohMiddlewarePubSub] Closing Zenoh session")
        self.session.close()

