import logging
import os
from glob import glob
from pathlib import Path


from wrapyfi.utils import SingletonOptimized, dynamic_module_import


class ListenerWatchDog(metaclass=SingletonOptimized):
    """
    A watchdog that scans for listeners and removes them from the ring if they are not established.
    """

    def __init__(self, repeats: int = 10, inner_repeats: int = 10):
        """
        Initialize the ListenerWatchDog.

        :param repeats: int: The number of times to repeat the scan
        param inner_repeats: int: The number of times to repeat the scan for each listener
        """
        self.repeats = repeats
        self.inner_repeats = inner_repeats
        self.listener_ring = []

    def add_listener(self, listener):
        """
        Add a listener to the ring.

        :param listener: Listener: The listener to add
        """
        self.listener_ring.append(listener)

    def remove_listener(self, listener):
        """
        Remove a listener from the ring.

        :param listener: Listener: The listener to remove
        """
        self.listener_ring.remove(listener)

    def scan(self):
        """
        Scan for listeners and remove them from the ring if they are established.
        """
        repeats = self.repeats
        while self.listener_ring and (repeats > 0 | repeats <= -1):
            repeats -= 1
            for listener in self.listener_ring:
                found_listener = listener.establish(repeats=self.inner_repeats)
                if found_listener:
                    self.listener_ring.remove(listener)


class Listeners(object):
    """
    A class that holds all listeners and their corresponding middleware communicators.
    """

    registry = {}
    mwares = set()

    @classmethod
    def register(cls, data_type: str, communicator: str):
        """
        Register a listener with the given data type and middleware communicator.

        :param data_type: str: The data type to register the listener for e.g., "NativeObject", "Image", "AudioChunk", etc.
        :param communicator: str: The middleware communicator to register the listener for e.g., "ros", "ros2", "yarp", "zeromq", etc.
        """

        def decorator(cls_):
            cls.registry[data_type + ":" + communicator] = cls_
            cls.mwares.add(communicator)
            return cls_

        return decorator

    @staticmethod
    def scan():
        """
        Scan for listeners and add them to the registry.
        """
        # modules = glob(
        #     os.path.join(os.path.dirname(__file__), "..", "listeners", "*.py"),
        #     recursive=True,
        # )
        # modules = [
        #     "wrapyfi.listeners."
        #     + module.replace(os.path.dirname(__file__) + "/../listeners/", "")
        #     for module in modules
        # ]
        # dynamic_module_import(modules, globals())
        base_dir = Path(__file__).parent.parent / "listeners"
        modules = glob(str(base_dir / "*.py"), recursive=True)

        modules = [
            "wrapyfi.listeners." + Path(module).relative_to(base_dir).as_posix()
            for module in modules
        ]
        dynamic_module_import(modules, globals())


class Listener(object):
    """
    A base class for listeners.
    """

    def __init__(
        self,
        name: str,
        in_topic: str,
        carrier: str = "",
        should_wait: bool = True,
        **kwargs,
    ):
        """
        Initialize the Listener.

        :param name: str: The name of the listener
        :param in_topic: str: The topic to listen to
        :param carrier: str: The middleware carrier to use
        :param should_wait: bool: Whether to wait for the listener to be established or not
        """
        self.__name__ = name
        self.in_topic = in_topic
        self.carrier = carrier
        self.should_wait = should_wait
        self.established = False

    def check_establishment(self, established: bool):
        """
        Check if the listener is established or not.

        :param established: bool: Whether the listener is established or not
        :return: bool: Whether the listener is established or not
        """
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

    def establish(self, repeats: int = -1, **kwargs):
        """
        Establish the listener.
        """
        raise NotImplementedError

    def listen(self):
        """
        Listen for incoming data.
        """
        raise NotImplementedError

    def close(self):
        """
        Close the connection.
        """
        raise NotImplementedError
