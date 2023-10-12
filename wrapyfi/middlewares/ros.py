import logging
import atexit

import rospy
import std_msgs.msg
import sensor_msgs.msg


from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class ROSMiddleware(metaclass=SingletonOptimized):
    """
    ROS middleware wrapper. This class is a singleton, so it can be instantiated only once. The ``activate`` method
    should be called to initialize the middleware. The ``deinit`` method should be called to deinitialize the middleware
    and destroy all connections. The ``activate`` and ``deinit`` methods are automatically called when the class is
    instantiated and when the program exits, respectively.
    """
    @staticmethod
    def activate(**kwargs):
        """
        Activate the ROS middleware. This method should be called to initialize the middleware.

        :param kwargs: dict: Keyword arguments to be passed to the ROS initialization function
        """
        ROSMiddleware(**kwargs)

    def __init__(self, node_name: str = "wrapyfi", anonymous: bool = True, disable_signals: bool = True, *args, **kwargs):
        """
        Initialize the ROS middleware. This method is automatically called when the class is instantiated.

        :param node_name: str: The name of the ROS node
        :param anonymous: bool: Whether the ROS node should be anonymous
        :param disable_signals: bool: Whether the ROS node should disable signals
        :param args: list: Positional arguments to be passed to the ROS initialization function
        :param kwargs: dict: Keyword arguments to be passed to the ROS initialization function
        """
        logging.info("Initialising ROS middleware")
        rospy.init_node(node_name, anonymous=anonymous, disable_signals=disable_signals)
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        """
        Deinitialize the ROS middleware. This method is automatically called when the program exits.
        """
        logging.info("Deinitializing ROS middleware")
        rospy.signal_shutdown('Deinit')


class ROSNativeObjectService(object):
  _type           = 'wrapyfi_services/ROSNativeObject'
  _md5sum         = '46a550fd1ca640b396e26ebf988aed7b'  # AddTwoInts '6a2e34150c00229791cc89ff309fff21'
  _request_class  = std_msgs.msg.String
  _response_class = std_msgs.msg.String


class ROSImageService(object):
  _type           = 'wrapyfi_services/ROSImage'
  _md5sum         = 'f720f2021b4bbbe86b0f93b08906381c'  # AddTwoInts '6a2e34150c00229791cc89ff309fff21'
  _request_class  = std_msgs.msg.String
  _response_class = sensor_msgs.msg.Image
