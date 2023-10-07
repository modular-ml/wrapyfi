"""
A message publisher and listener for native python objects and Dask DataFrames or Arrays

Here we demonstrate:
1. Using the NativeObject message
2. Transmitting a nested dummy python object with native objects and Dask DataFrames or Arrays
3. Applying the PUB/SUB pattern with mirroring

Requirements:
1. Install the dask package: pip install dask[complete]
2. Install the pandas (v1) package: pip install pandas<2.0

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 example_dask.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 example_dask.py --mode listen
"""

import argparse
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import numpy as np

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


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

        @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_dask_exchange",
                                         carrier="tcp", should_wait=True)
        def exchange_object(self):
            msg = input("Type your message: ")

            # Creating an example Dask DataFrame
            df = pd.DataFrame({
                'num_legs': [4, 2, 0, 4],
                'num_wings': [0, 2, 0, 0],
                'num_specimen_seen': [10, 2, 1, 8]
            }, index=['falcon', 'parrot', 'fish', 'dog'])
            ddf = dd.from_pandas(df, npartitions=2)

            # Creating an example Dask Array
            darray = da.random.random((1000, 1000), chunks=(250, 250))

            ret = [{"message": msg,
                    "dask_dataframe": ddf,
                    "dask_array": darray,
                    "list": [1, 2, 3]},
                   "string",
                   0.4344,
                   {"other": (1, 2, 3, 4.32,)}]
            return ret,


    notify = Notify()

    notify.activate_communication(Notify.exchange_object, mode=args.mode)
    while True:
        msg_object, = notify.exchange_object()

        # Compute and print the actual values of the Dask objects
        for item in msg_object:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, (dd.DataFrame, da.Array)):
                        print(f"{key} computed: \n{value.compute()}")
                    else:
                        print(f"{key}: {value}")
            else:
                print("Method result:", item)