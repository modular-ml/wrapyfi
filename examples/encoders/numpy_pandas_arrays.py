import argparse
import numpy as np
import pandas as pd

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

"""
A message publisher and listener for native python objects, numpy arrays and pandas dataframes

Here we demonstrate 
1. Using the NativeObject message
2. Transmit a nested dummy python object with native objects and multidim numpy arrays

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 numpy_pandas_arrays.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 numpy_pandas_arrays.py --mode listen

"""


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
            ret = [{"message": msg,
                    "numpy_array": np.ones((2, 4)),
                    "pandas_series": pd.Series([1, 3, 5, np.nan, 6, 8]),
                    "pandas_dataframe": pd.DataFrame(np.random.randn(6, 4), index=pd.date_range("20130101", periods=6),
                                                     columns=list("ABCD")),
                    "set": {'a', 1, None},
                    "list": [[[3, [4], 5.677890, 1.2]]]},
                   "string",
                   "string2",
                   0.4344,
                   {"other": (1, 2, 3, 4.32, )}]
            return ret,

    notify = Notify()

    notify.activate_communication(Notify.exchange_object, mode=args.mode)
    while True:
        msg_object, = notify.exchange_object()
        print("Method result:", msg_object)
