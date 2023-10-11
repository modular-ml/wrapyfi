"""
A message publisher and listener for ROS properties using Wrapyfi.

This script demonstrates the capability to transmit ROS properties, specifically
using the Properties message, using the MiddlewareCommunicator within the
Wrapyfi library. The communication follows the PUB/SUB pattern allowing
property publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the Properties message
    - Transmitting various data types as properties, such as strings, dictionaries, and non-persistent properties
    - Applying the PUB/SUB pattern with channeling and mirroring

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - ROS, ROSPy: Used for handling and creating ROS properties (refer to the Wrapyfi documentation for installation instructions)

    Ensure ROS is installed and the required message types are available.

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard input and transmits properties
        ``python3 ros_parameter_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for properties and prints the received properties
        ``python3 ros_parameter_example.py --mode listen``
"""

import argparse

from wrapyfi.connect.wrapper import MiddlewareCommunicator


class Notifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "Properties", "ros", "Notifier", "/notify/test_property_exchange",
        should_wait=True
    )
    @MiddlewareCommunicator.register(
        "Properties", "ros", "Notifier", "/notify/test_property_exchange/a",
        should_wait=True
    )
    @MiddlewareCommunicator.register(
        "Properties", "ros", "Notifier", "/notify/test_property_exchange/e",
        persistent=False, should_wait=True
    )
    def set_property(self):
        """Exchange ROS properties over ROS."""
        ret_str = input("Type your message: ")
        ret_multiprops = {"b": [1,2,3,4], "c": False, "d": 12.3}
        ret_non_persistent = "Non-persistent property which should be deleted on closure"
        return ret_multiprops, ret_str, ret_non_persistent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A message publisher and listener for ROS properties using Wrapyfi.")
    parser.add_argument(
        "--mode", type=str, default="publish",
        choices={"publish", "listen"},
        help="The transmission mode"
    )
    return parser.parse_args()


def main(args):
    """Main function to initiate Notify class and communication."""
    ros_message = Notifier()
    ros_message.activate_communication(Notifier.set_property, mode=args.mode)

    while True:
        my_dict_message, my_string_message, my_nonpersistent_message = ros_message.set_property()
        print("Method result:", my_string_message, my_dict_message, my_nonpersistent_message)


if __name__ == "__main__":
    args = parse_args()
    main(args)
