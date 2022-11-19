import logging
import atexit

import rclpy

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class ROS2Middleware(metaclass=SingletonOptimized):

    @staticmethod
    def activate(**kwargs):
        ROS2Middleware(**kwargs)

    def __init__(self, *args, **kwargs):
        logging.info("Initialising ROS 2 middleware")
        rclpy.init(args=[*args], **kwargs)
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        logging.info("Deinitialising ROS 2 middleware")
        rclpy.try_shutdown()
