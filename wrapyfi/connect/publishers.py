import logging
import os
from glob import glob
from pathlib import Path

from wrapyfi.utils import SingletonOptimized, dynamic_module_import, scan_external, WRAPYFI_MWARE_PATHS


class PublisherWatchDog(metaclass=SingletonOptimized):
    """
    A watchdog that scans for publishers and removes them from the ring if they are not established.
    """

    def __init__(self, repeats: int = 10, inner_repeats: int = 10):
        """
        Initialize the PublisherWatchDog.

        :param repeats: int: The number of times to repeat the scan
        param inner_repeats: int: The number of times to repeat the scan for each publisher
        """
        self.repeats = repeats
        self.inner_repeats = inner_repeats
        self.publisher_ring = []

    def add_publisher(self, publisher):
        """
        Add a publisher to the ring.

        :param publisher: Publisher: The publisher to add
        """
        self.publisher_ring.append(publisher)

    def remove_publisher(self, publisher):
        """
        Remove a publisher from the ring.

        :param publisher: Publisher: The publisher to remove
        """
        self.publisher_ring.remove(publisher)

    def scan(self):
        """
        Scan for publishers and remove them from the ring if they are established.
        """
        repeats = self.repeats
        while self.publisher_ring and (repeats > 0 | repeats <= -1):
            repeats -= 1
            for publisher in self.publisher_ring:
                found_publisher = publisher.establish(repeats=self.inner_repeats)
                if found_publisher:
                    self.publisher_ring.remove(publisher)


class Publishers(object):
    """
    A class that holds all publishers and their corresponding middleware communicators.
    """

    registry = {}
    mwares = set()

    @classmethod
    def register(cls, data_type: str, communicator: str):
        """
        Register a publisher for a given data type and middleware communicator.

        :param data_type: str: The data type to register the publisher for e.g., "NativeObject", "Image", "AudioChunk", etc.
        :param communicator: str: The middleware communicator to register the publisher for e.g., "ros", "ros2", "yarp", "zeromq", etc.
        :return: Callable[..., Any]: A decorator function that registers the decorated class as a publisher for the given data type and middleware communicator
        """

        def decorator(cls_):
            cls.registry[data_type + ":" + communicator] = cls_
            cls.mwares.add(communicator)
            return cls_

        return decorator

    @staticmethod
    def scan():
        """
        Scan for publishers and add them to the registry.
        """
        base_dir = Path(__file__).parent.parent / "publishers"
        modules = glob(str(base_dir / "*.py"), recursive=True)

        modules = [
            "wrapyfi.publishers." + Path(module).relative_to(base_dir).as_posix()
            for module in modules
        ]
        dynamic_module_import(modules, globals())
        scan_external(os.environ.get(WRAPYFI_MWARE_PATHS, ""), "publishers")


class Publisher(object):
    """
    A base class for all publishers.
    """

    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: str = "",
        should_wait: bool = True,
        **kwargs,
    ):
        """
        Initialize the Publisher.

        :param name: str: The name of the publisher
        :param out_topic: str: The name of the output topic
        :param carrier: str: The name of the carrier to use
        :param should_wait: bool: Whether to wait for the publisher to be established or not
        """
        self.__name__ = name
        self.out_topic = out_topic
        self.carrier = carrier
        self.should_wait = should_wait
        self.established = False

    def check_establishment(self, established: bool):
        """
        Check if the publisher is established and remove it from the ring if it is.

        :param established: bool: Whether the publisher is established or not
        :return: bool: Whether the publisher is established or not
        """
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

    def establish(self, repeats: int = -1, **kwargs):
        """
        Establish the publisher.
        """
        raise NotImplementedError

    def publish(self, obj):
        """
        Publish an object.
        """
        raise NotImplementedError

    def close(self):
        """
        Close the connection.
        """
        raise NotImplementedError
