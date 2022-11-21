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


# TODO (fabawi): Support multiple instance publishing of the same class,
#  currently only an issue with the output port naming convention
class Server(object):
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

    def reply(self, obj):
        raise NotImplementedError

    def await_request(self, *args, **kwargs):
        raise NotImplementedError
