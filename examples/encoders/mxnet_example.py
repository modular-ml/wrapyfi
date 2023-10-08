"""
A message publisher and listener for native Python objects and MXNet tensors.

This script demonstrates the capability to transmit native Python objects and MXNet tensors using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern
allowing message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a nested dummy Python object with native objects and MXNet tensors
    - Applying the PUB/SUB pattern with mirroring
    - Transmitting MXNet tensors with different contexts (CPU and GPU if available)
    - Flipping of contexts by mapping CPU to GPU and vice versa

Requirements:
    - Wrapyfi: Middleware communication wrapper (Refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS2, ZeroMQ (Refer to the Wrapyfi documentation for installation instructions)
    - MXNet: Used for handling and creating tensors (Refer to https://mxnet.apache.org/get_started?version=v1.8.0&platform=platform1&language=python&processor=cpu&environ=build_from_source&build=build for installation instructions)

    Install using pip:
        ``pip install mxnet``  # Basic installation of MXNet

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard input and transmits message
        ``python3 mxnet_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
        ``python3 mxnet_example.py --mode listen``
"""

import argparse

import mxnet

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class Notifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject", "$mware", "Notifier", "/notify/test_mxnet_exchange",
        carrier="", should_wait=True,
        listener_kwargs=dict(load_mxnet_device=mxnet.gpu(0), map_mxnet_devices={'cpu': 'cuda:0', 'gpu:0': 'cpu'})
    )
    def exchange_object(self, mware=None):
        """Exchange messages with MXNet tensors and other native Python objects."""
        msg = input("Type your message: ")
        ret = {
            "message": msg,
            "mx_ones": mxnet.nd.ones((2, 4), ctx=mxnet.cpu()),
            "mxnet_zeros_cuda": mxnet.nd.zeros((2, 3), ctx=mxnet.gpu(0))
        }
        return ret,


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A message publisher and listener for native Python objects and MXNet tensors.")
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
    """Main function to initiate Notifier class and communication."""
    notifier = Notifier()
    notifier.activate_communication(Notifier.exchange_object, mode=args.mode)

    while True:
        msg_object, = notifier.exchange_object(mware=args.mware)
        print("Method result:", msg_object)


if __name__ == "__main__":
    args = parse_args()
    main(args)
