import atexit
import rospy

from wrapify.utils import SingletonOptimized


class ROSMiddleware(metaclass=SingletonOptimized):

    @staticmethod
    def activate():
        ROSMiddleware()

    def __init__(self):
        print("Initialising ROS middleware")
        rospy.init_node('wrapify', anonymous=True, disable_signals=True)
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        print("Deinitialising ROS middleware")
        rospy.signal_shutdown('Deinit')
