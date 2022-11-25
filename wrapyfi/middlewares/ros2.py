import logging
import atexit

import rclpy
from std_msgs.msg._string import Metaclass_String
from sensor_msgs.msg._image import Metaclass_Image

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator


# TODO (fabawi): Not functional yet. Custom messages must be compiled first
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


class Metaclass_ROS2NativeObjectService(type):
    """Metaclass of service 'ROS2NativeObjectService'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('std_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'wrapyfi.middleware.ros2.ROS2NativeObjectService')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_msg__msg__string
            # cls._TYPE_SUPPORT = module.type_support_srv__srv__trigger
            if Metaclass_String._TYPE_SUPPORT is None:
                Metaclass_String.__import_type_support__()
            # if Metaclass_Image._TYPE_SUPPORT is None:
            #     Metaclass_Image.__import_type_support__()


class ROS2NativeObjectService(metaclass=Metaclass_ROS2NativeObjectService):
    from std_msgs.msg._string import String as Request
    from std_msgs.msg._string import String as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
