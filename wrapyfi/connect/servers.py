import logging
import os
from glob import glob

from wrapyfi.utils import SingletonOptimized, dynamic_module_import


class Servers(object):
    registry = {}
    mwares = set()

    @classmethod
    def register(cls, data_type, communicator):
        def decorator(cls_):
            cls.registry[data_type + ":" + communicator] = cls_
            cls.mwares.add(communicator)
            return cls_
        return decorator

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "servers", "*.py"), recursive=True)
        modules = ["wrapyfi.servers." + module.replace(os.path.dirname(__file__) + "/../servers/", "") for module in
                   modules]
        dynamic_module_import(modules, globals())


class Server(object):
    def __init__(self, name, out_topic, carrier="", out_topic_connect=None, **kwargs):
        self.__name__ = name
        self.out_topic = out_topic
        self.carrier = carrier
        self.out_topic_connect = out_topic + ":out" if out_topic_connect is None else out_topic_connect
        self.established = False

    def establish(self):
        raise NotImplementedError

    def await_request(self, *args, **kwargs):
        raise NotImplementedError

    def reply(self, obj):
        raise NotImplementedError
