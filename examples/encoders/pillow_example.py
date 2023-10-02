import argparse

import numpy as np
from PIL import Image

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

"""
A message publisher and listener for pillow images

Here we demonstrate
1. Using the NativeObject message
2. Transmit a nested dummy python object with native objects and pillow images

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 pillow_image.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 pillow_image.py --mode listen

"""


class Notify(MiddlewareCommunicator):
    @MiddlewareCommunicator.register("NativeObject", "$mware", "Notify", "/notify/test_native_exchange",
                                     carrier="", should_wait=True)
    def exchange_object(self, mware=None):
        msg = input("Type your message: ")
        imarray = np.random.rand(100, 100, 3) * 255
        ret = {"message": msg,
               "pillow_random": Image.fromarray(imarray.astype('uint8')).convert('RGBA'),
               "pillow_png": Image.open("../../resources/wrapyfi.png"),
               "pillow_jpg": Image.open("../../resources/wrapyfi.jpg")}
        return ret,

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"}, help="The transmission mode")
    parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                        help="The middleware to use for transmission")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    notify = Notify()

    notify.activate_communication(Notify.exchange_object, mode=args.mode)

    while True:
        msg_object, = notify.exchange_object(mware=args.mware)
        print("Method result:", msg_object)
