"""
Mirroring Example using Wrapyfi.

This script demonstrates the capability to mirror messages using the MiddlewareCommunicator within the Wrapyfi library.
The communication follows the PUB/SUB and REQ/REP patterns, allowing message publishing, listening, requesting, and replying
functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a Python object with user input and predefined message
    - Applying the PUB/SUB and REQ/REP patterns with mirroring

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)

Run:
    # Alternative 1: PUB/SUB
        # On machine 1 (or process 1): Publishing mode - Publisher waits for keyboard input and transmits message

        ``python3 mirroring_example.py --mode publish``

        # On machine 2 (or process 2): Listening mode - Listener waits for message and prints the received object

        ``python3 mirroring_example.py --mode listen``

    # Alternative 2: REQ/REP
        # On machine 1 (or process 1): Replying mode - Replier waits for a message, sends a reply, and prints the received object

        ``python3 mirroring_example.py --mode reply``

        # On machine 2 (or process 2): Requesting mode - Requester sends a predefined message and waits for a reply

        ``python3 mirroring_example.py --mode request``
"""

import argparse

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class MirrorCls(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        'NativeObject', '$mware', 'MirrorCls',
        '/example/read_msg',
        carrier='tcp', should_wait='$blocking')
    def read_msg(self, mware=None, msg='', blocking=True):
        """Exchange messages and mirror user input."""
        msg_ip = input('Type message: ')
        obj = {'msg': msg, 'msg_ip': msg_ip}
        return obj,

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mirroring Example using Wrapyfi.")
    parser.add_argument(
        "--mode", type=str, default="listen",
        choices={"publish", "listen", "request", "reply"},
        help="The communication mode (publish, listen, request, reply)"
    )
    parser.add_argument(
        "--mware", type=str, default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use for transmission"
    )
    return parser.parse_args()

def main(args):
    """Main function to initiate MirrorCls class and communication."""
    mirror = MirrorCls()
    mirror.activate_communication(MirrorCls.read_msg, mode=args.mode)

    while True:
        msg_object, = mirror.read_msg(mware=args.mware, msg=f"This argument message was sent by the {args.mode} script")
        if msg_object is not None:
            print(msg_object)


if __name__ == "__main__":
    args = parse_args()
    main(args)
