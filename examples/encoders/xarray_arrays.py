import argparse
import numpy as np
import xarray as xr
import pandas as pd

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

"""
A message publisher and listener for native python objects and xarray DataArrays or Datasets.

Here we demonstrate:
1. Using the NativeObject message
2. Transmit a nested dummy python object with native objects and xarray DataArrays or Datasets.

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 example_xarray.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 example_xarray.py --mode listen
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

        @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_xarray_exchange",
                                         carrier="tcp", should_wait=True)
        def exchange_object(self):
            msg = input("Type your message: ")

            # Creating an example xarray DataArray
            data = np.random.rand(4, 3)
            locs = ['IA', 'IL', 'IN']
            times = pd.date_range('2000-01-01', periods=4)
            da = xr.DataArray(data, coords=[times, locs], dims=['time', 'space'], name='example')

            ret = [{"message": msg,
                    "xarray_dataarray": da,
                    "set": {'a', 1, None},
                    "list": [[[3, [4], 5.677890, 1.2]]]},
                   "string",
                   "string2",
                   0.4344,
                   {"other": (1, 2, 3, 4.32,)}]
            return ret,


    notify = Notify()

    notify.activate_communication(Notify.exchange_object, mode=args.mode)
    while True:
        msg_object, = notify.exchange_object()
        print("Method result:", msg_object)