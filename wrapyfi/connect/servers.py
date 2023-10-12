import logging
import os
from glob import glob
from typing import Optional

from wrapyfi.utils import dynamic_module_import


class Servers(object):
    """
    A class that holds all servers and their corresponding middleware communicators.
    """
    registry = {}
    mwares = set()

    @classmethod
    def register(cls, data_type: str, communicator: str):
        """
        Register a server with the given data type and middleware communicator.

        :param data_type: str: The data type to register the server for e.g., "NativeObject", "Image", "AudioChunk", etc.
        :param communicator: str: The middleware communicator to register the server for e.g., "ros", "ros2", "yarp", "zeromq", etc.
        :return: Callable: A decorator that registers the server with the given data type and middleware communicator
        """
        def decorator(cls_):
            cls.registry[data_type + ":" + communicator] = cls_
            cls.mwares.add(communicator)
            return cls_
        return decorator

    @staticmethod
    def scan():
        """
        Scan for servers and add them to the registry.
        """
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "servers", "*.py"), recursive=True)
        modules = ["wrapyfi.servers." + module.replace(os.path.dirname(__file__) + "/../servers/", "") for module in
                   modules]
        dynamic_module_import(modules, globals())


class Server(object):
    """
    A base class for servers.
    """
    def __init__(self, name: str, out_topic: str, carrier: str = "", out_topic_connect: Optional[str] = None, **kwargs):
        """
        Initialize the server.

        :param name: str: The name of the server
        :param out_topic: str: The topic to publish to
        :param carrier: str: The middleware carrier to use
        :param out_topic_connect: str: The topic to connect to (this is deprecated and will be removed in the future since its usage is limited to YARP)
        """
        self.__name__ = name
        self.out_topic = out_topic
        self.carrier = carrier
        self.out_topic_connect = out_topic + ":out" if out_topic_connect is None else out_topic_connect
        self.established = False

    def establish(self):
        """
        Establish the server.
        """
        raise NotImplementedError

    def await_request(self, *args, **kwargs):
        """
        Await a request from a client.
        """
        raise NotImplementedError

    def reply(self, obj):
        """
        Reply to a client request.
        """
        raise NotImplementedError

    def close(self):
        """
        Close the connection.
        """
        raise NotImplementedError
