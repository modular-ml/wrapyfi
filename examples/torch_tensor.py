import argparse
import torch

from wrapify.connect.wrapper import MiddlewareCommunicator

"""
A message publisher and listener for torch tensors

Here we demonstrate
1. Using the NativeObject message
2. Transmit a nested dummy python object with native objects and multidim torch tensors
3. Demonstrating the responsive transmission

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message. (First message is ignored)
    python3 torch_tensor.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 torch_tensor.py --mode listen

"""

class Notify(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", "yarp", "Notify", "/notify/test_native_bottle_exchange", carrier="", should_wait=True, load_torch_device='cpu')
    def exchange_object(self, msg):
        ret = [{"message": msg,
                "tensor": torch.ones((2, 4), device='cpu'),
                "list": [[[3, 4, 5.677890, 1.2]]]}, "some", "arbitrary", 0.4344, {"other": torch.zeros((2, 3), device='cuda')}]
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
