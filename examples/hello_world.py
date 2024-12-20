"""
This example shows how to use the MiddlewareCommunicator to send and receive messages. It can be used to test the
functionality of the middleware using the PUB/SUB pattern and the REQ/REP pattern. The example can be run on a single
machine or on multiple machines. In this example (as with all other examples), the communication middleware is selected
using the ``--mware`` argument. The default is ZeroMQ, but YARP, ROS, and ROS 2 are also supported.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS 2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)

Run:
    # Alternative 1: PUB/SUB mode
        # On machine 1 (or process 1): PUB/SUB mode - Publisher waits for keyboard input and transmits message

        ``python3 hello_world.py --publish --mware zeromq``

        # On machine 2 (or process 2): PUB/SUB mode - Listener waits for message and prints the received object

        ``python3 hello_world.py --listen --mware zeromq``

    # Alternative 2: REQ/REP mode
        # On machine 1 (or process 1): REQ/REP mode - Replier waits for a message, sends a reply, and prints the received object

        ``python3 hello_world.py --reply --mware zeromq``

        # On machine 2 (or process 2): REQ/REP mode - Requester sends a predefined message and waits for a reply

        ``python3 hello_world.py --request --mware zeromq``


"""

import argparse

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class HelloWorld(MiddlewareCommunicator):

    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware",
        "HelloWorld",
        "/hello/my_message",
        carrier="tcp",
        should_wait=True,
    )
    def send_message(self, arg_from_requester="", mware=None):
        """
        Exchange messages and mirror user input.
        """
        msg = input("Type your message: ")
        obj = {"message": msg, "message_from_requester": arg_from_requester}
        return (obj,)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publish",
        dest="mode",
        action="store_const",
        const="publish",
        default="listen",
        help="Publish mode",
    )
    parser.add_argument(
        "--listen",
        dest="mode",
        action="store_const",
        const="listen",
        default="listen",
        help="Listen mode (default)",
    )
    parser.add_argument(
        "--transceive",
        dest="mode",
        action="store_const",
        const="transceive",
        default="listen",
        help="Transceive mode - publish the method and listen for output instead of just returning published output",
    )
    parser.add_argument(
        "--request",
        dest="mode",
        action="store_const",
        const="request",
        default="listen",
        help="Request mode",
    )
    parser.add_argument(
        "--reply",
        dest="mode",
        action="store_const",
        const="reply",
        default="listen",
        help="Reply mode",
    )
    parser.add_argument(
        "--mware",
        type=str,
        default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use for transmission",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hello_world = HelloWorld()
    hello_world.activate_communication(HelloWorld.send_message, mode=args.mode)
    while True:
        (my_message,) = hello_world.send_message(
            arg_from_requester=f"I got this message from the script running in {args.mode} mode",
            mware=args.mware,
        )
        if my_message is not None:
            print("Method result:", my_message)
