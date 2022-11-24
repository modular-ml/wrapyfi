import argparse
from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

parser = argparse.ArgumentParser()
parser.add_argument("--request", dest="mode", action="store_const", const="request", default="request", help="Transmit arguments and await reply")
parser.add_argument("--reply", dest="mode", action="store_const", const="reply", default="request", help="Wait for request and return results/reply")
parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                    help="The middleware to use for transmission")
args = parser.parse_args()


class ReqRep(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", args.mware, "ReqRep", "/req_rep/my_message",
                                     carrier="tcp")
    def send_message(self, *args, **kwargs):
        msg = input("Type your message: ")
        obj = {"message": msg,
               "args": args,
               "kwargs": kwargs}
        return obj,


req_rep = ReqRep()
req_rep.activate_communication(ReqRep.send_message, mode=args.mode)

counter = 0
while True:
    if args.mode == "request":
        my_message, = req_rep.send_message(counter=counter)
        counter += 1
        if my_message is not None:
            print("Request: counter:", counter)
            print("Request: received reply:", my_message)
    if args.mode == "reply":
        my_message, = req_rep.send_message()
        if my_message is not None:
            print("Reply: received reply:", my_message)