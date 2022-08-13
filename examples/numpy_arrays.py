import argparse
import numpy as np

from wrapify.connect.wrapper import MiddlewareCommunicator

"""
A message publisher and listener for native python objects and numpy arrays

Here we demonstrate 
1. Using the NativeObject message
2. Transmit a nested dummy python object with native objects and multidim numpy arrays
3. Demonstrating the responsive transmission

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 numpy_arrays.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 numpy_arrays.py --mode listen

"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"}, help="The transmission mode")
    parser.add_argument("--mware", type=str, default="yarp", choices={"yarp", "ros"}, help="The middleware to use for transmission")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    class Notify(MiddlewareCommunicator):

        @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_native_exchange", carrier="", should_wait=True)
        def exchange_object(self, msg):
            ret = [{"message": msg,
                    "numpy": np.ones((2, 4)),
                    "set": {'a', 1, None},
                    "list": [[[3, 4, 5.677890, 1.2]]]}, "some", "arbitrary", 0.4344, {"other": np.zeros((2, 3))}]
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
