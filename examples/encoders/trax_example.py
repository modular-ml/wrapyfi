"""
A message publisher and listener for native Python objects and Trax arrays.

This script demonstrates the capability to transmit native Python objects and Trax arrays using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern
allowing message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a nested dummy Python object with native objects and Trax arrays
    - Applying the PUB/SUB pattern with mirroring

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS 2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - Trax: Used for handling Google Trax arrays (refer to https://trax-ml.readthedocs.io/en/latest/notebooks/trax_intro.html for installation instructions)

    Install using pip:
        ``pip install trax``

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard input and transmits message
        ``python3 trax_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
        ``python3 trax_example.py --mode listen``
"""

import argparse

try:
    import trax
    from trax.fastmath import numpy as fastnp
    trax.fastmath.use_backend('tensorflow-numpy')
except ImportError:
    print("Install Trax before running this script.")

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class Notifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware",
        "Notifier",
        "/notify/test_trax_exchange",
        carrier="tcp",
        should_wait=True,
    )
    def exchange_object(self, mware=None):
        """
        Exchange messages with Trax arrays and other native Python objects.
        """
        msg = input("Type your message: ")
        ret = {
            "message": msg,
            "trax_array": fastnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "trax_ones": fastnp.ones(3),
            "trax_zeros": fastnp.zeros(3),
        }
        return (ret,)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A message publisher and listener for native Python objects and Trax arrays."
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
