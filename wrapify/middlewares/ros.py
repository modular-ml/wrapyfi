import logging
import atexit

import rospy

from wrapify.utils import SingletonOptimized


class ROSMiddleware(metaclass=SingletonOptimized):

    @staticmethod
    def activate(**kwargs):
        ROSMiddleware(**kwargs)

    def __init__(self, *args, **kwargs):
        logging.info("Initialising ROS middleware")
        rospy.init_node('wrapify', anonymous=True, disable_signals=True)
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        logging.info("Deinitialising ROS middleware")
        rospy.signal_shutdown('Deinit')
