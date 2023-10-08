"""
A message publisher and listener for native Python objects and PaddlePaddle tensors.

This script demonstrates the capability to transmit native Python objects and PaddlePaddle tensors using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern
allowing message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a nested dummy Python object with native objects and PaddlePaddle tensors
    - Applying the PUB/SUB pattern with mirroring
    - Transmitting PaddlePaddle tensors with different devices (CPU and GPU if available)
    - Flipping of devices by mapping CPU to GPU and vice versa

Requirements:
    - Wrapyfi: Middleware communication wrapper (Refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS2, ZeroMQ (Refer to the Wrapyfi documentation for installation instructions)
    - PaddlePaddle: Used for handling and creating tensors (Refer to https://www.paddlepaddle.org.cn/en/install/quick for installation instructions)

    Install using pip:
        ``pip install paddlepaddle-gpu``  # Basic installation of PaddlePaddle

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard input and transmits message
        ``python3 paddlepaddle_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
        ``python3 paddlepaddle_example.py --mode listen``
"""

import argparse

import paddle

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class Notifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject", "$mware", "Notifier", "/notify/test_paddle_exchange",
        carrier="", should_wait=True,
        listener_kwargs=dict(load_paddle_device='gpu:0', map_paddle_devices={'cpu': 'cuda:0', 'gpu:0': 'cpu'})
    )
    def exchange_object(self, mware=None):
        """Exchange messages with PaddlePaddle tensors and other native Python objects."""
        msg = input("Type your message: ")
        ret = {
            "message": msg,
            "paddle_ones": paddle.ones([2, 4], dtype='float32', place=paddle.CPUPlace()),
            "paddle_zeros_cuda": paddle.zeros([2, 3], dtype='float32', place=paddle.CUDAPlace(0))
        }
        return ret,


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A message publisher and listener for native Python objects and PaddlePaddle tensors.")
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
