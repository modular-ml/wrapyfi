import os
from glob import glob

from wrapyfi.utils import SingletonOptimized, dynamic_module_import


class ListenerWatchDog(metaclass=SingletonOptimized):
    def __init__(self, repeats=10, inner_repeats=10):
        self.repeats = repeats
        self.inner_repeats = inner_repeats
        self.listener_ring = []

    def add_listener(self, listener):
        self.listener_ring.append(listener)

    def remove_listener(self, listener):
        self.listener_ring.remove(listener)

    def scan(self):
        repeats = self.repeats
        while self.listener_ring and (repeats > 0 | repeats <= -1):
            repeats -= 1
            for listener in self.listener_ring:
                found_listener = listener.establish(repeats=self.inner_repeats)
                if found_listener:
                    self.listener_ring.remove(listener)


class Listeners(object):
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
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "listeners", "*.py"), recursive=True)
        modules = ["wrapyfi.listeners." + module.replace(os.path.dirname(__file__) + "/../listeners/", "") for module in
                   modules]
        dynamic_module_import(modules, globals())


class Listener(object):
    def __init__(self, name, in_port, carrier="", should_wait=True, **kwargs):

        self.__name__ = name
        self.in_port = in_port
        self.carrier = carrier
        self.should_wait = should_wait
        self.established = False

    def check_establishment(self, established):
        if established:
            self.established = True
            if not self.should_wait:
                ListenerWatchDog().remove_listener(self)
        elif not self.should_wait:
            ListenerWatchDog().scan()
            if self in ListenerWatchDog().listener_ring:
                established = False
            else:
                established = True
        return established

    def establish(self, repeats=-1, **kwargs):
        raise NotImplementedError

    def listen(self):
        raise NotImplementedError
