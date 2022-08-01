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
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message. (First message is ignored)
    python3 object_notify.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 object_notify.py --mode listen

"""

class Notify(MiddlewareCommunicator):
    @MiddlewareCommunicator.register("NativeObject", "yarp", "Notify", "/notify/test_native_bottle_exchange", carrier="", should_wait=True)
    def exchange_object(self, msg):
        ret = [{"message": msg,
                "numpy": np.ones((2, 4)),
                "list": [[[3, 4, 5.677890, 1.2]]]}, "some", "arbitrary", 0.4344, {"other": np.zeros((2, 3))}]
        return ret,

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"}, help="The transmission mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    notify = Notify()

    if args.mode == "publish":
        notify.activate_communication(Notify.exchange_object, mode="publish")
        while True:
            notify.exchange_object(input("Type your message: "))
    elif args.mode == "listen":
        notify.activate_communication(Notify.exchange_object, mode="listen")
        while True:
            obj = notify.exchange_object(None)
            if obj and obj[0] is not None:
                print(obj)
