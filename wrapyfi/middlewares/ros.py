import logging
import atexit

import rospy
import std_msgs.msg
import sensor_msgs.msg


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


class ROSNativeObjectService(object):
  _type          = 'wrapyfi_services/ROSService'
  _md5sum = '46a550fd1ca640b396e26ebf988aed7b'  # AddTwoInts '6a2e34150c00229791cc89ff309fff21'
  _request_class  = std_msgs.msg.String
  _response_class = std_msgs.msg.String

