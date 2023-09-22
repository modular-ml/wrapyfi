import logging
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


@Clients.register("MMO", "fallback")
class FallbackClient(Client):

    def __init__(self, name: str, in_port: str, carrier: str = "", missing_middleware_object: str = "", **kwargs):
        logging.warning(f"Fallback client employed due to missing middleware or object type: "
                        f"{missing_middleware_object}")
        Client.__init__(self, name, in_port, carrier=carrier, **kwargs)
        self.missing_middleware_object = missing_middleware_object

    def establish(self, repeats=-1, **kwargs):
        return None

    def request(self, *args, **kwargs):
        return None

    def _request(self, *args, **kwargs):
        return None

    def _await_reply(self):
        return None

    def close(self):
        return None