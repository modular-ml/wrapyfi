"""
A message publisher and listener for native Python objects and Dask Arrays/Dataframes.

This script demonstrates the capability to transmit native Python objects and Dask arrays/dataframes using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern
allowing message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a nested dummy Python object with native objects and Dask arrays/dataframes
    - Applying the PUB/SUB pattern with mirroring

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - Dask, pandas: Used for handling and creating arrays and dataframes

    Install using pip:
        ``pip install "pandas<2.0" dask[complete]``

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard input and transmits message
        ``python3 dask_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for message and computes and prints the Dask objects
        ``python3 dask_example.py --mode listen``
"""

import argparse

try:
    import dask.array as da
    import dask.dataframe as dd
    import pandas as pd
except ImportError:
    print("Install DASK and pandas before running this script.")

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class Notifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject", "$mware", "Notifier", "/notify/test_dask_exchange",
        carrier="tcp", should_wait=True
    )
    def exchange_object(self, mware=None):
        """Exchange messages with Dask arrays/dataframes and other native Python objects."""
        msg = input("Type your message: ")

        # Creating an example Dask DataFrame
        df = pd.DataFrame({
            'num_legs': [4, 2, 0, 4],
            'num_wings': [0, 2, 0, 0],
            'num_specimen_seen': [10, 2, 1, 8]
        }, index=['falcon', 'parrot', 'fish', 'dog'])

        ddf = dd.from_pandas(df, npartitions=2)

        ds = pd.Series([1, 2, 3, 4, 5, 6, 7])
        dds = dd.from_pandas(ds, npartitions=1)
        # Creating an example Dask Array
        darray = da.random.random((1000, 1000), chunks=(250, 250))

        ret = {
            "message": msg,
            "dask_dataframe": ddf,
            "dask_array": darray,
            "dask_series": dds,
        }
        return ret,


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A message publisher and listener for native Python objects and Dask arrays/dataframes.")
    parser.add_argument(
        "--mode", type=str, default="publish",
        choices={"publish", "listen"},
        help="The transmission mode"
    )
    parser.add_argument(
        "--mware", type=str, default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use for transmission"
    )
    return parser.parse_args()


def main(args):
    """Main function to initiate Notifier class and communication."""
    notifier = Notifier()
    notifier.activate_communication(Notifier.exchange_object, mode=args.mode)

    while True:
        msg_object, = notifier.exchange_object(mware=args.mware)

        # Compute and print the actual values of the Dask objects
        for key, value in msg_object.items():
            if isinstance(value, (dd.DataFrame, dd.Series, da.Array)):
                print(f"{key} computed: \n{value.compute()}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
