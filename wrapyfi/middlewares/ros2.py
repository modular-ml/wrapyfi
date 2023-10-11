import logging
import atexit

import rclpy

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class ROS2Middleware(metaclass=SingletonOptimized):
    """
    ROS2 middleware wrapper. This class is a singleton, so it can be instantiated only once. The ``activate`` method
    should be called to initialise the middleware. The ``deinit`` method should be called to deinitialise the middleware
    and destroy all connections. The ``activate`` and ``deinit`` methods are automatically called when the class is
    instantiated and when the program exits, respectively.
    """

    @staticmethod
    def activate(**kwargs):
        """
        Activate the ROS2 middleware. This method should be called to initialise the middleware.

        :param kwargs: dict: Keyword arguments to be passed to the ROS2 initialisation function
        """
        ROS2Middleware(**kwargs)

    def __init__(self, *args, **kwargs):
        """
        Initialise the ROS2 middleware. This method is automatically called when the class is instantiated.

        :param args: list: Positional arguments to be passed to the ROS2 initialisation function
        :param kwargs: dict: Keyword arguments to be passed to the ROS2 initialisation function
        """
        logging.info("Initialising ROS2 middleware")
        rclpy.init(args=[*args], **kwargs)
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        """
        Deinitialise the ROS2 middleware. This method is automatically called when the program exits.
        """
        logging.info("Deinitialising ROS2 middleware")
        rclpy.try_shutdown()

