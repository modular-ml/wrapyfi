import argparse
from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

parser = argparse.ArgumentParser()
parser.add_argument("--mode_chain_A", type=str, default="publish", choices=["listen", "publish",
                                                                           "disable", "none", None],
                    help="The mode of transmission for the first method in the chain")
parser.add_argument("--mode_chain_B", type=str, default="listen", choices=["listen", "publish",
                                                                           "disable", "none", None],
                    help="The mode of transmission for the second method in the chain")
parser.add_argument("--mware_chain_A", type=str, default=DEFAULT_COMMUNICATOR,
                    choices=MiddlewareCommunicator.get_communicators(),
                    help="The middleware to use for transmission of the first method in the chain")
parser.add_argument("--mware_chain_B", type=str, default=DEFAULT_COMMUNICATOR,
                    choices=MiddlewareCommunicator.get_communicators(),
                    help="The middleware to use for transmission of the second method in the chain")
args = parser.parse_args()


class ForwardCls(MiddlewareCommunicator):
    @MiddlewareCommunicator.register('NativeObject',
                                     args.mware_chain_A, 'ForwardCls', '/example/native_chain_A_msg',
                                     carrier='mcast', should_wait=True)
    def read_chain_A(self, msg):
        return msg,

    @MiddlewareCommunicator.register('NativeObject',
                                     args.mware_chain_B, 'ForwardCls', '/example/native_chain_B_msg',
                                     carrier='tcp')
    def read_chain_B(self, msg):
        return msg,


forward = ForwardCls()
forward.activate_communication(forward.read_chain_A, mode=args.mode_chain_A)
forward.activate_communication(forward.read_chain_B, mode=args.mode_chain_B)

while True:
    msg, = forward.read_chain_A(f"this argument message was sent from read_chain_A trainsmitted over "
                                f"{args.mware_chain_A}")
    if msg is not None:
        print(msg)
    msg, = forward.read_chain_B(f"{msg}. It was then forwarded to read_chain_B over "
                                f"{args.mware_chain_B}")
    if msg is not None:
        if args.mode_chain_B == "listen":
            print(f"{msg}. This message is the last in the chain received over {args.mware_chain_B}")
        else:
            print(msg)
