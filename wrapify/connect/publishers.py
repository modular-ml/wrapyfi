import os
from glob import glob

from wrapify.utils import SingletonOptimized, dynamic_module_import


class Publishers(object):
    registry = {}
    mwares = set()

    @classmethod
    def register(cls, data_type, communicator):
        def decorator(klass):
            cls.registry[data_type + ":" + communicator] = klass
            cls.mwares.add(communicator)
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
    def __init__(self, repeats=10, inner_repeats=10):
        self.repeats = repeats
        self.inner_repeats = inner_repeats
        self.publisher_ring = []

    def add_publisher(self, publisher):
        self.publisher_ring.append(publisher)

    def remove_publisher(self, publisher):
        self.publisher_ring.remove(publisher)

    def scan(self):
        repeats = self.repeats
        while self.publisher_ring and (repeats > 0 | repeats <= -1):
            repeats -= 1
            for publisher in self.publisher_ring:
                found_publisher= publisher.establish(repeats=self.inner_repeats)
                if found_publisher:
                    self.publisher_ring.remove(publisher)


# TODO (fabawi): Support multiple instance publishing of the same class,
#  currently only an issue with the output port naming convention
class Publisher(object):
    def __init__(self, name, out_port, carrier="", out_port_connect=None, should_wait=True, **kwargs):
        self.__name__ = name
        self.out_port = out_port
        self.carrier = carrier
        self.out_port_connect = out_port + ":out" if out_port_connect is None else out_port_connect
        self.should_wait = should_wait
        self.established = False

    def check_establishment(self, established):
        if established:
            self.established = True
            if not self.should_wait:
                PublisherWatchDog().remove_publisher(self)
        elif not self.should_wait:
            PublisherWatchDog().scan()
            if self in PublisherWatchDog().publisher_ring:
                established = False
            else:
                established = True
        return established

    def establish(self, repeats=-1, **kwargs):
        raise NotImplementedError

    def publish(self, obj):
        raise NotImplementedError

