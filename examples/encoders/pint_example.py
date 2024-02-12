"""
A message publisher and listener for native Python objects and Pint Quantities.

This script demonstrates the capability to transmit native Python objects and Pint Quantities using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern
allowing message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a nested dummy Python object with native objects and Pint Quantities
    - Applying the PUB/SUB pattern with mirroring

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS 2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - Pint: Used for handling physical quantities with units

    Install using pip:
        ``pip install pint``

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard input and transmits message
        ``python3 pint_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
        ``python3 pint_example.py --mode listen``
"""

import argparse

try:
    import pint
except ImportError:
    print("Install Pint before running this script.")

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class Notifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware",
        "Notifier",
        "/notify/test_pint_exchange",
        carrier="tcp",
        should_wait=True,
    )
    def exchange_object(self, mware=None):
        """
        Exchange messages with Pint Quantities and other native Python objects.
        """
        msg = input("Type your message: ")

        # Creating a Pint Quantity
        ureg = pint.UnitRegistry()
        quantity = 42 * ureg.parse_expression("meter")

        # Constructing the message object to be transmitted
        ret = [
            {"message": msg, "pint_quantity": quantity, "list": [1, 2, 3]},
            "string",
            0.4344,
            {
                "other": (
                    1,
                    2,
                    3,
                    4.32,
                )
            },
        ]
        return (ret,)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A message publisher and listener for native Python objects and Pint Quantities."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="publish",
        choices={"publish", "listen"},
        help="The transmission mode",
    )
    parser.add_argument(
        "--mware",
        type=str,
        default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use for transmission",
    )
    return parser.parse_args()


def main(args):
    """
    Main function to initiate Notifier class and communication.
    """
    notifier = Notifier()
    notifier.activate_communication(Notifier.exchange_object, mode=args.mode)

    while True:
        (msg_object,) = notifier.exchange_object(mware=args.mware)
        print("Method result:", msg_object)


if __name__ == "__main__":
    args = parse_args()
    main(args)
