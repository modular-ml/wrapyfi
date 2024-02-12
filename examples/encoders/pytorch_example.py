"""
A message publisher and listener for native Python objects and PyTorch tensors.

This script demonstrates the capability to transmit native Python objects and PyTorch tensors using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern,
allowing message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a nested dummy Python object with native objects and PyTorch tensors
    - Applying the PUB/SUB pattern with mirroring
    - Transmitting PyTorch tensors with different devices (CPU and GPU if available)
    - Flipping of devices by mapping CPU to GPU and vice versa

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS 2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - PyTorch: Used for handling and creating tensors (refer to https://pytorch.org/get-started/locally/ for installation instructions)

    Install using pip:
        ``pip install "torch>=1.12.1"``  # Basic installation of PyTorch

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard input and transmits message
        ``python3 pytorch_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
        ``python3 pytorch_example.py --mode listen``
"""

import argparse

try:
    import torch
except ImportError:
    print("Install PyTorch before running this script.")

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class Notifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware",
        "Notifier",
        "/notify/test_torch_exchange",
        carrier="",
        should_wait=True,
        listener_kwargs=dict(
            load_torch_device="cuda:0",
            map_torch_devices={"cpu": "cuda:0", "cuda:0": "cpu"},
        ),
    )
    def exchange_object(self, mware=None):
        """
        Exchange messages with PyTorch tensors and other native Python objects.
        """
        msg = input("Type your message: ")
        ret = {
            "message": msg,
            "torch_ones": torch.ones((2, 4), device="cpu"),
            "torch_zeros_cuda": torch.zeros((2, 3), device="cuda:0"),
        }
        return (ret,)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A message publisher and listener for native Python objects and PyTorch tensors."
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
