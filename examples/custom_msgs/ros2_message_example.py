"""
A message publisher and listener for ROS2 messages using Wrapyfi.

This script demonstrates the capability to transmit ROS2 messages, specifically
geometry_msgs/Pose and std_msgs/String, using the MiddlewareCommunicator within
the Wrapyfi library. The communication follows the PUB/SUB pattern allowing
message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the ROS2 message
    - Transmitting std_msgs/String and geometry_msgs/Pose ROS2 messages
    - Applying the PUB/SUB pattern with channeling and mirroring

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - ROS2, rclpy: Used for handling and creating ROS2 messages (refer to the Wrapyfi documentation for installation instructions)

    Ensure ROS2 is installed and the required message types are available.

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard input and transmits message
        ``python3 ros2_message_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for message and prints the received ROS2 messages
        ``python3 ros2_message_example.py --mode listen``
"""

import argparse

from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import String

from wrapyfi.connect.wrapper import MiddlewareCommunicator


class Notifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "ROS2Message", "ros2", "Notifier", "/notify/test_ros2_msg_str_exchange",
        should_wait=True
    )
    @MiddlewareCommunicator.register(
        "ROS2Message", "ros2", "Notifier", "/notify/test_ros2_msg_pose_exchange",
        should_wait=True
    )
    def send_message(self):
        """Exchange ROS2 messages over ROS2."""
        msg = input("Type your message: ")
        quat = Quaternion()
        quat.x = 0.1
        quat.y = -0.1
        quat.z = 0.8
        quat.z = -0.8
        ret_str = String(data=msg)
        ret_pose = Pose()
        ret_pose.orientation = quat
        return ret_str, ret_pose


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A message publisher and listener for ROS 2 messages using Wrapyfi.")
    parser.add_argument(
        "--mode", type=str, default="publish",
        choices={"publish", "listen"},
        help="The transmission mode"
    )
    return parser.parse_args()


def main(args):
    """Main function to initiate Notify class and communication."""
    ros2_message = Notifier()
    ros2_message.activate_communication(Notifier.send_message, mode=args.mode)

    while True:
        my_string_message, my_pose_message = ros2_message.send_message()
        if my_string_message is not None or my_pose_message is not None:
            print("Method result:", my_string_message, my_pose_message)


if __name__ == "__main__":
    args = parse_args()
    main(args)
