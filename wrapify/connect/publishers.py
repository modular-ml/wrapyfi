import os
from glob import glob

from wrapify.utils import SingletonOptimized, dynamic_module_import


class Publishers(object):
    registry = {}

    @classmethod
    def register(cls, data_type, communicator):
        def decorator(klass):
            cls.registry[data_type + ":" + communicator] = klass
            return klass
        return decorator

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "publishers", "*.py"), recursive=True)
        modules = ["wrapify.publishers." + module.replace(os.path.dirname(__file__) + "/../publishers/", "") for module in
                   modules]
        dynamic_module_import(modules, globals())


# TODO (fabawi): The watch dog is not running yet. Relying on lazy publishing for now
class PublisherWatchDog(metaclass=SingletonOptimized):
    def __init__(self):
        self.publisher_ring = []

    def add_publisher(self, publisher):
        self.publisher_ring.append(publisher)

    def scan(self):
        while self.publisher_ring:
            for publisher in self.publisher_ring:
                found_listener = publisher.establish()
                if found_listener:
                    self.publisher_ring.remove(publisher)


# TODO (fabawi): Support multiple instance publishing of the same class,
#  currently only an issue with the output port naming convention
class Publisher(object):
    def __init__(self, name, out_port, carrier="", out_port_connect=None, **kwargs):
        self.__name__ = name
        self.out_port = out_port
        self.carrier = carrier
        self.out_port_connect = out_port + ":out" if out_port_connect is None else out_port_connect

        self.established = False

    def establish(self, **kwargs):
        raise NotImplementedError

    def publish(self, obj):
        raise NotImplementedError

