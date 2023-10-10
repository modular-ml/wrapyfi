"""
Forwarding Example using Wrapyfi.

This script demonstrates message forwarding using the MiddlewareCommunicator within the Wrapyfi library.
The communication follows chained forwarding through two methods, enabling PUB/SUB pattern
that allows message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a Python object with predefined message
    - Applying the PUB/SUB pattern with forwarding through two communication methods

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)

Run:
    # On machine 1 (or process 1): Mode Chain A: Publisher [ZeroMQ], Chain B: Disabled (chain A & B can have different middleware
    by setting the --mware_... argument but it should be consistent on all machines or processes)
        ``python3 forwarding_example.py --mware_chain_A zeromq --mode_chain_A publish --mode_chain_B disable``

    # On machine 2 (or process 2): Mode Chain A: Listener [ZeroMQ], Chain B: Publisher [ROS2] (chain A & B can have different middleware
    by setting the --mware_... argument but it should be consistent on all machines or processes)
        ``python3 forwarding_example.py --mware_chain_A zeromq --mode_chain_A listen --mware_chain_B ros2 --mode_chain_B publish``

    # On machine 3 (or process 3): Mode Chain A: Disabled, Chain B: Listener [ROS2] (chain A & B can have different middleware
    by setting the --mware_... argument but it should be consistent on all machines or processes)
        ``python3 forwarding_example.py --mode_chain_A disable --mware_chain_B ros2 --mode_chain_B listen``
"""

import time
import argparse

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class ForwardCls(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        'NativeObject', '$mware_chain_A', 'ForwardCls', '/example/native_chain_A_msg',
        carrier='mcast', should_wait=True)
    def read_chain_A(self, mware_chain_A=None, msg=''):
        """Read and forward message from chain A."""
        return msg,

    @MiddlewareCommunicator.register(
        'NativeObject', '$mware_chain_B', 'ForwardCls', '/example/native_chain_B_msg',
        carrier='tcp', should_wait=False)
    def read_chain_B(self, mware_chain_B=None, msg=''):
        """Read and forward message from chain B."""
        return msg,

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Forwarding Example using Wrapyfi.")
    parser.add_argument(
        "--mode_chain_A", type=str, default="publish",
        choices=["listen", "publish", "disable", "none", None],
        help="The mode of transmission for the first method in the chain"
    )
    parser.add_argument(
        "--mode_chain_B", type=str, default="listen",
        choices=["listen", "publish", "disable", "none", None],
        help="The mode of transmission for the second method in the chain"
    )
    parser.add_argument(
        "--mware_chain_A", type=str, default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use for transmission of the first method in the chain"
    )
    parser.add_argument(
        "--mware_chain_B", type=str, default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use for transmission of the second method in the chain"
    )
    return parser.parse_args()


def main(args):
    """Main function to initiate ForwardCls class and communication."""
    forward = ForwardCls()
    forward.activate_communication(forward.read_chain_A, mode=args.mode_chain_A)
    forward.activate_communication(forward.read_chain_B, mode=args.mode_chain_B)

    while True:
        msg, = forward.read_chain_A(mware_chain_A=args.mware_chain_A,
                                    msg=f"This argument message was sent from read_chain_A transmitted over "
                                       f"{args.mware_chain_A}")
        if msg is not None:
            print(msg)
        msg, = forward.read_chain_B(mware_chain_B=args.mware_chain_B,
                                    msg=f"{msg}. It was then forwarded to read_chain_B over {args.mware_chain_B}")
        if msg is not None:
            if args.mode_chain_B == "listen":
                print(f"{msg}. This message is the last in the chain received over {args.mware_chain_B}")
            else:
                print(msg)
        time.sleep(0.1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
