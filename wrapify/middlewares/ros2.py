import logging
import atexit

import rclpy

from wrapify.utils import SingletonOptimized


class ROS2Middleware(metaclass=SingletonOptimized):

    @staticmethod
    def activate(**kwargs):
        ROS2Middleware(**kwargs)

    def __init__(self, *args, **kwargs):
        logging.info("Initialising ROS 2 middleware")
        rclpy.init(args=[*args], **kwargs)
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        logging.info("Deinitialising ROS 2 middleware")
        rclpy.try_shutdown()
