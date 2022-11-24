import os
from glob import glob

from wrapyfi.utils import dynamic_module_import


class Clients(object):
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
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "clients", "*.py"), recursive=True)
        modules = ["wrapyfi.clients." + module.replace(os.path.dirname(__file__) + "/../clients/", "") for module in
                   modules]
        dynamic_module_import(modules, globals())


class Client(object):
    def __init__(self, name, in_port, carrier="", **kwargs):
        self.__name__ = name
        self.in_port = in_port
        self.carrier = carrier
        self.established = False

    def establish(self):
        raise NotImplementedError

    def request(self, *args, **kwargs):
        raise NotImplementedError

    def _request(self, *args, **kwargs):
        raise NotImplementedError

    def _await_reply(self):
        raise NotImplementedError