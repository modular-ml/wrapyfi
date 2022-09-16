import argparse
from wrapify.connect.wrapper import MiddlewareCommunicator
import torch
import tensorflow
import mxnet

parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")
parser.add_argument("--mware", type=str, default="yarp", choices={"yarp", "ros"}, help="The middleware to use for transmission")
args = parser.parse_args()


class ExampleClass(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", "yarp",
                                     "ExampleClass", "/example/read_message",
                                     carrier="tcp", should_wait=True)
    def read_message(self):
        msg = input("Type your message: ")
        obj = {"message": msg}
        return obj,


hello_world = ExampleClass()
hello_world.activate_communication(ExampleClass.read_message, mode=args.mode)

while True:
    my_message, = hello_world.read_message()
    print("Method result:", my_message["message"])
