"""
A message publisher and listener for PIL (Pillow) images.

This script demonstrates the capability to transmit native Python objects and Pillow images in different formats using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern
allowing message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a nested dummy Python object with native objects and PIL (Pillow) images
    - Applying the PUB/SUB pattern with mirroring

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - NumPy: Used for creating image arrays (installed with Wrapyfi)
    - PIL (Pillow): Used for handling image objects

    Install using pip:
        ``pip install pillow``

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
        ``python3 pillow_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
        ``python3 pillow_example.py --mode listen``
"""

import argparse

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("Install PIL (Pillow) and NumPy before running this script.")

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class Notifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject", "$mware", "Notify", "/notify/test_native_exchange",
        carrier="", should_wait=True
    )
    def exchange_object(self, mware=None):
        msg = input("Type your message: ")
        imarray = np.random.rand(100, 100, 3) * 255
        ret = {
            "message": msg,
            "pillow_random": Image.fromarray(imarray.astype('uint8')).convert('RGBA'),
            "pillow_png": Image.open("../../resources/wrapyfi.png"),
            "pillow_jpg": Image.open("../../resources/wrapyfi.jpg")
        }
        return ret,


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="publish",
        choices={"publish", "listen"},
        help="The transmission mode"
    )
    parser.add_argument(
        "--mware", type=str, default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use for transmission"
    )
    return parser.parse_args()


def main(args):
    """Main function to initiate Notify class and communication."""
    notifier = Notifier()
    notifier.activate_communication(Notifier.exchange_object, mode=args.mode)

    while True:
        msg_object, = notifier.exchange_object(mware=args.mware)
        print("Method result:", msg_object)


if __name__ == "__main__":
    args = parse_args()
    main(args)
