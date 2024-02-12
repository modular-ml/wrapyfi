"""
A message publisher and listener for native Python objects and CuPy arrays.

This script demonstrates the capability to transmit native Python objects and CuPy arrays using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern
allowing message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a nested dummy Python object with native objects and CuPy arrays
    - Applying the PUB/SUB pattern with mirroring
    - Transmitting CuPy arrays across different devices (only GPU devices are supported)

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS 2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - CuPy: Used for handling and creating arrays with GPU acceleration (refer to https://docs.cupy.dev/en/stable/install.html for installation instructions)

    Install using pip:
        ``pip install cupy-cuda12x``  # Basic installation of CuPy. Replace 12x with your CUDA version e.g., cupy-cuda11x

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard input and transmits message
        ``python3 cupy_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
        ``python3 cupy_example.py --mode listen``
"""

import argparse

try:
    import cupy as cp
except ImportError:
    print("Install CuPy before running this script.")

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class CuPyNotifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware",
        "CuPyNotifier",
        "/notify/test_cupy_exchange",
        carrier="",
        should_wait=True,
        listener_kwargs=dict(load_cupy_device=cp.cuda.Device(0)),
    )
    def exchange_object(self, mware=None):
        """
        Exchange messages with CuPy tensors and other native Python objects.
        """
        msg = input("Type your message: ")
        ret = {
            "message": msg,
            "cupy_ones_cuda": cp.ones((2, 4), dtype=cp.float32),
            "cupy_zeros_cuda": cp.zeros((2, 3), dtype=cp.float32),
        }
        return (ret,)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A message publisher and listener for native Python objects and CuPy tensors."
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
    Main function to initiate CuPyNotifier class and communication.
    """
    notifier = CuPyNotifier()
    notifier.activate_communication(CuPyNotifier.exchange_object, mode=args.mode)

    while True:
        (msg_object,) = notifier.exchange_object(mware=args.mware)
        print("Method result:", msg_object)


if __name__ == "__main__":
    args = parse_args()
    main(args)
