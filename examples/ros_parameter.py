import argparse

from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import String

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")

args = parser.parse_args()


class Notify(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("Properties", "ros", "Notify", "/notify/test_property_exchange", should_wait=False)
    @MiddlewareCommunicator.register("Properties", "ros", "Notify", "/notify/test_property_exchange/a", should_wait=False)
    @MiddlewareCommunicator.register("Properties", "ros", "Notify", "/notify/test_property_exchange/e", persistent=False, should_wait=False)
    def set_property(self):
        ret_str = input("Type your message: ")
        ret_multiprops = {"b": [1,2,3,4], "c": False, "d": 12.3}
        ret_non_persistent = "Non-persistent property which should be deleted on closure"
        return ret_multiprops, ret_str, ret_non_persistent

ros_message = Notify()
ros_message.activate_communication(Notify.set_property, mode=args.mode)

while True:
    my_dict_message, my_string_message, my_nonpersistent_message = ros_message.set_property()
    if my_string_message is not None or my_dict_message is not None or my_nonpersistent_message is not None:
        print("Method result:", my_string_message, my_dict_message, my_nonpersistent_message)
