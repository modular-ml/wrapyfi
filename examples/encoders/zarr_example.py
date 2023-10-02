import argparse
import zarr
import numpy as np

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

"""
A message publisher and listener for native python objects and Zarr Arrays or Groups.

Here we demonstrate:
1. Using the NativeObject message
2. Transmit a nested dummy python object with native objects and Zarr Arrays or Groups.

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 example_zarr.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 example_zarr.py --mode listen
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"},
                        help="The transmission mode")
    parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR,
                        choices=MiddlewareCommunicator.get_communicators(),
                        help="The middleware to use for transmission")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()


    class Notify(MiddlewareCommunicator):

        @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_zarr_exchange",
                                         carrier="tcp", should_wait=True)
        def exchange_object(self):
            msg = input("Type your message: ")

            # Creating an example Zarr Array
            zarray = zarr.array(np.random.random((10, 10)), chunks=(5, 5))

            # Creating an example Zarr Group
            zgroup = zarr.group()
            zgroup.create_dataset('dataset1', data=np.random.randint(0, 100, 50), chunks=10)
            zgroup.create_dataset('dataset2', data=np.random.random(100), chunks=10)

            ret = [{"message": msg,
                    "zarr_array": zarray,
                    "zarr_group": zgroup,
                    "list": [1, 2, 3]},
                   "string",
                   0.4344,
                   {"other": (1, 2, 3, 4.32,)}]
            return ret,


    notify = Notify()

    notify.activate_communication(Notify.exchange_object, mode=args.mode)
    while True:
        msg_object, = notify.exchange_object()
        print("Method result:", msg_object)
