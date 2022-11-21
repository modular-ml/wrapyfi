import os
from glob import glob

from wrapyfi.utils import SingletonOptimized, dynamic_module_import


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
    def __init__(self, name, out_port, carrier="", out_port_connect=None, **kwargs):
        self.__name__ = name
        self.out_port = out_port
        self.carrier = carrier
        self.out_port_connect = out_port + ":out" if out_port_connect is None else out_port_connect
        self.established = False

    def check_establishment(self, established):
        if established:
            self.established = True
        return established

    def establish(self, repeats=-1, **kwargs):
        raise NotImplementedError

    def request(self, **kwargs):
        raise NotImplementedError

    def await_reply(self, obj):
        raise NotImplementedError