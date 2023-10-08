import argparse
from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")
parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                    help="The middleware to use for transmission")
args = parser.parse_args()


class HelloWorld(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", args.mware, "HelloWorld", "/hello/my_message",
                                     carrier="tcp", should_wait=True)
    def send_message(self):
        msg = input("Type your message: ")
        obj = {"message": msg}
        return obj,


hello_world = HelloWorld()
hello_world.activate_communication(HelloWorld.send_message, mode=args.mode)

while True:
    my_message, = hello_world.send_message()
    if my_message is not None:
        print("Method result:", my_message["message"])
