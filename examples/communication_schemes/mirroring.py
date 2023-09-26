import argparse
from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")
parser.add_argument("--request", dest="mode", action="store_const", const="request", default="listen", help="Request mode")
parser.add_argument("--reply", dest="mode", action="store_const", const="reply", default="listen", help="Reply mode")
parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                    help="The middleware to use for transmission")
args = parser.parse_args()


class MirrorCls(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        'NativeObject', '$0', 'MirrorCls',
        '/example/read_msg',
        carrier='tcp', should_wait='$blocking')
    def read_msg(self, mware, msg='', blocking=True):
        msg_ip = input('type message:')
        obj = {'msg': msg, 'msg_ip': msg_ip}
        return obj,


mirror = MirrorCls()
mirror.activate_communication('read_msg', mode=args.mode)

while True:
    msg, = mirror.read_msg(args.mware, f"this argument message was sent by the {args.mode} script")
    if msg is not None:
        print(msg)
