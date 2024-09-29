"""
This example shows how to use the MiddlewareCommunicator to send and receive messages over Websockets. It can be used to test the
functionality of the Websockets using the PUB/SUB pattern. The example can be run on a single
machine or on multiple machines. In this example (as with all other examples), the listener awaits a message over the topic '/hello/my_message'.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - Websockets (refer to the Wrapyfi documentation for installation instructions)

Run:
    # PUB/SUB mode - Listener waits for message and prints the received object (assuming the websocket server is running). Only one instance of the websocket_server.py should be running

        ``python3 listener_client.py``

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
    def receive_message(self):
        """
        Exchange messages and mirror user input.
        """
        return (None,)


if __name__ == "__main__":
    hello_world = HelloWorld()
    hello_world.activate_communication(HelloWorld.receive_message, mode="listen")

    while True:
        (my_message,) = hello_world.receive_message()
        if my_message is not None:
            print("Method result:", my_message)
