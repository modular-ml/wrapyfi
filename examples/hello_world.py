import argparse
from wrapify.connect.wrapper import MiddlewareCommunicator


class HelloWorld(MiddlewareCommunicator):
    @MiddlewareCommunicator.register("NativeObject", "yarp", "HelloWorld", "/hello/my_message",
                                     carrier="", should_wait=True)
    def send_message(self):
        msg = input("Type your message: ")
        obj = {"message": msg}
        return obj,


parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")
args = parser.parse_args()

hello_world = HelloWorld()
hello_world.activate_communication(HelloWorld.send_message, mode=args.mode)

while True:
    my_message, = hello_world.send_message()
    print(my_message["message"])
