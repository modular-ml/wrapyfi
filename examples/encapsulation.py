import argparse
from wrapify.connect.wrapper import MiddlewareCommunicator


class Encapsulator(MiddlewareCommunicator):
    @MiddlewareCommunicator.register("NativeObject", "yarp", "Encapsulator", "/encapsulator/my_message_modifier",
                                     carrier="", should_wait=True)
    def encapsulated_modify_message(self, msg):
        msg = "******" + msg + "******"
        return msg,

    @MiddlewareCommunicator.register("NativeObject", "yarp", "Encapsulator", "/encapsulator/my_message",
                                     carrier="", should_wait=True)
    def encapsulating_send_message(self):
        msg = input("Type your message: ")
        msg, = self.encapsulated_modify_message(msg)
        obj = {"message": msg}
        return obj,


parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")
args = parser.parse_args()

encapsulator = Encapsulator()
encapsulator.activate_communication("encapsulating_send_message", mode=args.mode)
encapsulator.activate_communication("encapsulated_modify_message", mode=args.mode)

while True:
    my_message, = encapsulator.encapsulating_send_message()
    print(my_message["message"])
