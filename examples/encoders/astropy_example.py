import os
script_dir = os.path.dirname(os.path.realpath(__file__))
# modify the WRAPYFI_PLUGINS_PATH environment variable
if 'WRAPYFI_PLUGINS_PATH' in os.environ:
    os.environ['WRAPYFI_PLUGINS_PATH'] += os.pathsep + script_dir
else:
    os.environ['WRAPYFI_PLUGINS_PATH'] = script_dir

import argparse
from astropy.table import Table
import numpy as np

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

"""
A message publisher and listener for native python objects and Astropy Tables.

Here we demonstrate:
1. Using the NativeObject message
2. Transmit a nested dummy python object with native objects and Astropy Tables
3. Use a plugin that is not installed by default (astropy)

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 astropy_example.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 astropy_example.py --mode listen
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

        @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_astropy_exchange",
                                         carrier="tcp", should_wait=True)
        def exchange_object(self):
            msg = input("Type your message: ")

            # Creating an example Astropy Table
            t = Table()
            t['name'] = ['source 1', 'source 2', 'source 3']
            t['flux'] = [1.2, 2.2, 3.1]

            ret = [{"message": msg,
                    "astropy_table": t,
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