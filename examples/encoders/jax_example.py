"""
A message publisher and listener for Google JAX tensors

Here we demonstrate:
1. Using the NativeObject message
2. Transmitting a nested dummy python object with native objects and multidim JAX tensors
3. Applying the PUB/SUB pattern with mirroring

Requirements:
1. Install the jax package: pip install jax

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 jax_tensor.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 jax_tensor.py --mode listen

"""

import argparse
import jax_example.numpy as jnp

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"}, help="The transmission mode")
    parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                        help="The middleware to use for transmission")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    class Notify(MiddlewareCommunicator):

        @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_native_exchange",
                                         carrier="tcp", should_wait=True)
        def exchange_object(self):
            msg = input("Type your message: ")
            ret = {"message": msg,
                   "jax_ones": jnp.ones((2, 4))}
            return ret,

    notify = Notify()

    notify.activate_communication(Notify.exchange_object, mode=args.mode)
    while True:
        msg_object, = notify.exchange_object()
        print("Method result:", msg_object)

