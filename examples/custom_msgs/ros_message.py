import argparse

from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import String

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")

args = parser.parse_args()


class Notify(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("ROSMessage", "ros", "Notify", "/notify/test_ros_msg_str_exchange", should_wait=False)
    @MiddlewareCommunicator.register("ROSMessage", "ros", "Notify", "/notify/test_ros_msg_pose_exchange", should_wait=False)
    def send_message(self):
        msg = input("Type your message: ")
        quat = Quaternion()
        quat.x = 0.1
        quat.y = -0.1
        quat.z = 0.8
        quat.z = -0.8
        ret_str = String(msg)
        ret_pose = Pose()
        ret_pose.orientation = quat
        return ret_str, ret_pose

ros_message = Notify()
ros_message.activate_communication(Notify.send_message, mode=args.mode)

while True:
    my_string_message, my_pose_message = ros_message.send_message()
    if my_string_message is not None or my_pose_message is not None:
        print("Method result:", my_string_message, my_pose_message)
