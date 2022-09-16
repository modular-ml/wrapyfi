import argparse
import mxnet

from wrapify.connect.wrapper import MiddlewareCommunicator

"""
A message publisher and listener for MXNet tensors

Here we demonstrate
1. Using the NativeObject message
2. Transmit a nested dummy python object with native objects and multidim MXNet tensors
3. Demonstrating the responsive transmission

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 mxnet_tensor.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 mxnet_tensor.py --mode listen

"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"}, help="The transmission mode")
    parser.add_argument("--mware", type=str, default="yarp", choices={"yarp", "ros", "zeromq"}, help="The middleware to use for transmission")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    class Notify(MiddlewareCommunicator):

        @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_native_exchange",
                                         carrier="", should_wait=True, load_mxnet_device=mxnet.gpu(0))
        def exchange_object(self, msg):
            ret = {"message": msg,
                   "mx_ones": mxnet.nd.ones((2, 4)),
                   "mxnet_zeros_cuda": mxnet.nd.zeros((2, 3), ctx=mxnet.gpu(0))}
            return ret,

    notify = Notify()

    if args.mode == "publish":
        notify.activate_communication(Notify.exchange_object, mode="publish")
        while True:
            msg_object, = notify.exchange_object(input("Type your message: "))
            print("Method result:", msg_object)
    elif args.mode == "listen":
        notify.activate_communication(Notify.exchange_object, mode="listen")
        while True:
            msg_object, = notify.exchange_object(None)
            print("Method result:", msg_object)
