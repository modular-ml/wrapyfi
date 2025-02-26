import logging
import os
from glob import glob
from pathlib import Path

from wrapyfi.utils.core_utils import dynamic_module_import, scan_external, WRAPYFI_MWARE_PATHS


class Clients(object):
    """
    A class that holds all clients and their corresponding middleware communicators.
    """

    registry = {}
    mwares = set()

    @classmethod
    def register(cls, data_type: str, communicator: str):
        """
        Register a client with the given data type and middleware communicator.

        :param data_type: str: The data type to register the client for e.g., "NativeObject", "Image", "AudioChunk", etc.
        :param communicator: str: The middleware communicator to register the client for e.g., "ros", "ros2", "yarp", "zeromq", etc.
        """

        def decorator(cls_):
            cls.registry[data_type + ":" + communicator] = cls_
            cls.mwares.add(communicator)
            return cls_

        return decorator

    @staticmethod
    def scan():
        """
        Scan for clients and add them to the registry.
        """
        base_dir = Path(__file__).parent.parent / "clients"
        modules = glob(str(base_dir / "*.py"), recursive=True)

        modules = [
            "wrapyfi.clients." + Path(module).relative_to(base_dir).as_posix()
            for module in modules
        ]
        dynamic_module_import(modules, globals())
        scan_external(os.environ.get(WRAPYFI_MWARE_PATHS, ""), "clients")


class Client(object):
    """
    A base class for clients.
    """

    def __init__(self, name: str, in_topic: str, carrier: str = "", **kwargs):
        """
        Initialize the client.

        :param name: str: The name of the client
        :param in_topic: str: The topic to listen to
        :param carrier: str: The middleware carrier to use
        """
        self.__name__ = name
        self.in_topic = in_topic
        self.carrier = carrier
        self.established = False

    def establish(self):
        """
        Establish the client.
        """
        raise NotImplementedError

    def request(self, *args, **kwargs):
        """
        Send a request to the server.
        """
        raise NotImplementedError

    def _request(self, *args, **kwargs):
        """
        Internal method for sending a request to the server in the background.
        """
        raise NotImplementedError

    def _await_reply(self):
        """
        Internal method for awaiting a reply from the server in the background.
        """
        raise NotImplementedError

    def close(self):
        """
        Close the connection.
        """
        raise NotImplementedError
