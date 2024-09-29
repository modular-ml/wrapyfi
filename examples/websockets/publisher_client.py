"""
This example shows how to use the MiddlewareCommunicator to send and receive messages over Websockets. It can be used to test the
functionality of the Websockets using the PUB/SUB pattern. The example can be run on a single
machine or on multiple machines. In this example (as with all other examples), the publisher transmits a message over the topic '/hello/my_message'.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - Websockets (refer to the Wrapyfi documentation for installation instructions)

Run:
    # PUB/SUB mode - Publisher transmits message and prints the received object (assuming the websocket server is running). Only one instance of the websocket_server.py should be running

        ``python3 publisher_client.py``

"""

import argparse

from wrapyfi.connect.wrapper import MiddlewareCommunicator


class HelloWorld(MiddlewareCommunicator):

    @MiddlewareCommunicator.register(
        "NativeObject",
        "websocket",
        "HelloWorld",
        "/hello/my_message",
        carrier="tcp",
        should_wait=True,
    )
    def send_message(self, arg_from_requester=""):
        """
        Exchange messages and mirror user input.
        """
        msg = input("Type your message: ")
        obj = {"message": msg, "message_from_requester": arg_from_requester}
        return (obj,)


if __name__ == "__main__":
    hello_world = HelloWorld()
    hello_world.activate_communication(HelloWorld.send_message, mode="publish")

    while True:
        (my_message,) = hello_world.send_message(
            arg_from_requester=f"I got this message from the script running in publisher mode",
        )
        if my_message is not None:
            print("Method result:", my_message)
