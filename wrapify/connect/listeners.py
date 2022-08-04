import os
from glob import glob

from wrapify.utils import SingletonOptimized, dynamic_module_import


# TODO (fabawi): The watch dog is not running yet. Relying on lazy listening for now
class ListenerWatchDog(metaclass=SingletonOptimized):
    def __init__(self):
        self.listener_ring = []

    def add_listener(self, listener):
        self.listener_ring.append(listener)

    def scan(self):
        while self.listener_ring:
            for listener in self.listener_ring:
                found_listener = listener.establish()
                if found_listener:
                    self.listener_ring.remove(listener)


class Listeners(object):
    registry = {}

    @classmethod
    def register(cls, data_type, communicator):
        def decorator(klass):
            cls.registry[data_type + ":" + communicator] = klass
            return klass
        return decorator

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "listeners", "*.py"), recursive=True)
        modules = ["wrapify.listeners." + module.replace(os.path.dirname(__file__) + "/../listeners/", "") for module in
                   modules]
        dynamic_module_import(modules, globals())


class Listener(object):
    def __init__(self, name, in_port, carrier="", should_wait=True):
        self.__name__ = name
        self.in_port = in_port
        self.carrier = carrier
        self.should_wait = should_wait

        self.established = False

    def establish(self, **kwargs):
        raise NotImplementedError

    def listen(self):
        raise NotImplementedError
