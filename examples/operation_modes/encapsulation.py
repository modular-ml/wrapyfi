import argparse
from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")
parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                    help="The middleware to use for transmission")
args = parser.parse_args()


class Encapsulator(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", args.mware, "Encapsulator", "/encapsulator/my_message_modifier", carrier="", should_wait=True)
    def encapsulated_modify_message(self, msg):
        return f"****** {msg} ******",

    @MiddlewareCommunicator.register("NativeObject", args.mware, "Encapsulator", "/encapsulator/my_message", carrier="", should_wait=True)
    def encapsulating_send_message(self):
        msg = input("Type your message: ")
        msg, = self.encapsulated_modify_message(msg)
        obj = {"message": msg}
        return obj,


encapsulator = Encapsulator()
encapsulator.activate_communication(Encapsulator.encapsulating_send_message, mode=args.mode)
encapsulator.activate_communication(Encapsulator.encapsulated_modify_message, mode=args.mode)

encapsulator.encapsulated_modify_message('')
while True:
    my_message, = encapsulator.encapsulating_send_message()
    print("Method result:", my_message["message"])
