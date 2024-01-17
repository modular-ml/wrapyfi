import logging
import atexit

import rclpy

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class ROS2Middleware(metaclass=SingletonOptimized):
    """
    ROS 2 middleware wrapper. This class is a singleton, so it can be instantiated only once. The ``activate`` method
    should be called to initialize the middleware. The ``deinit`` method should be called to deinitialize the middleware
    and destroy all connections. The ``activate`` and ``deinit`` methods are automatically called when the class is
    instantiated and when the program exits, respectively.
    """

    @staticmethod
    def activate(**kwargs):
        """
        Activate the ROS 2 middleware. This method should be called to initialize the middleware.

        :param kwargs: dict: Keyword arguments to be passed to the ROS 2 initialization function
        """
        ROS2Middleware(**kwargs)

    def __init__(self, *args, **kwargs):
        """
        Initialize the ROS 2 middleware. This method is automatically called when the class is instantiated.

        :param args: list: Positional arguments to be passed to the ROS 2 initialization function
        :param kwargs: dict: Keyword arguments to be passed to the ROS 2 initialization function
        """
        logging.info("Initialising ROS 2 middleware")
        rclpy.init(args=[*args], **kwargs)
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        """
        Deinitialize the ROS 2 middleware. This method is automatically called when the program exits.
        """
        logging.info("Deinitializing ROS 2 middleware")
        rclpy.shutdown()

