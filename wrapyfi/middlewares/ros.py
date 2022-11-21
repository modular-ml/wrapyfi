import logging
import atexit

import rospy

from wrapyfi.utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class ROSMiddleware(metaclass=SingletonOptimized):

    @staticmethod
    def activate(**kwargs):
        ROSMiddleware(**kwargs)

    def __init__(self, node_name="wrapyfi", anonymous=True, disable_signals=True, *args, **kwargs):
        logging.info("Initialising ROS middleware")
        rospy.init_node(node_name, anonymous=anonymous, disable_signals=disable_signals)
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        logging.info("Deinitialising ROS middleware")
        rospy.signal_shutdown('Deinit')
