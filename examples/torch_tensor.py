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
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 torch_tensor.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 torch_tensor.py --mode listen

"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"}, help="The transmission mode")
    parser.add_argument("--mware", type=str, default="yarp", choices={"yarp", "ros"}, help="The middleware to use for transmission")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    class Notify(MiddlewareCommunicator):

        @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_native_exchange",
                                         carrier="", should_wait=False, load_torch_device='cpu')
        @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_native_exchange2",
                                         carrier="", should_wait=False, load_torch_device='cpu')
        def exchange_object(self, msg):
            ret = {"message": msg,
                   "torch_ones": torch.ones((2, 4), device='cpu'),
                   "torch_zeros_cuda": torch.zeros((2, 3), device='cuda')}
            str_ret = f"this is your message transmitted as a string {msg}"
            return ret, str_ret

    notify = Notify()

    if args.mode == "publish":
        notify.activate_communication(Notify.exchange_object, mode="publish")
        while True:
            msg_object, msg_str = notify.exchange_object(input("Type your message: "))
            print("Method result:", msg_object, msg_str)
    elif args.mode == "listen":
        notify.activate_communication(Notify.exchange_object, mode="listen")
        while True:
            msg_object, msg_str = notify.exchange_object(None)
            if msg_object is not None:
                print("Method result:", msg_object, msg_str)
