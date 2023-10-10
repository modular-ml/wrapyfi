"""
A message publisher and listener for native Python objects and Astropy Tables (external plugin).

This script demonstrates the capability to transmit native Python objects and Astropy Tables using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern
allowing message publishing and listening functionalities between processes or machines. By modifying the
``WRAPYFI_PLUGINS_PATH`` environment variable to include the current directory, which contains the
plugins directory with ``astropy_tables.py`` file, the ``AstropyTable`` message can be used to transmit Astropy Tables.

Demonstrations:
    - Using the NativeObject message
    - Transmitting a nested dummy Python object with native objects and Astropy Tables
    - Applying the PUB/SUB pattern with mirroring
    - Using an external plugin to handle Astropy Tables

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - Astropy: Used for creating and handling astronomical tables

    Install using pip:
        ``pip install astropy``

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard input and transmits message
        ``python3 astropy_example.py --mode publish``

    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
        ``python3 astropy_example.py --mode listen``
"""

import os
import argparse

from astropy.table import Table

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

# Modifying the WRAPYFI_PLUGINS_PATH environment variable to include the plugins directory
script_dir = os.path.dirname(os.path.realpath(__file__))
if 'WRAPYFI_PLUGINS_PATH' in os.environ:
    os.environ['WRAPYFI_PLUGINS_PATH'] += os.pathsep + script_dir
else:
    os.environ['WRAPYFI_PLUGINS_PATH'] = script_dir


class Notifier(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject", "$mware", "Notifier", "/notify/test_astropy_exchange",
        carrier="tcp", should_wait=True
    )
    def exchange_object(self, mware=None):
        """Exchange messages with Astropy Tables and other native Python objects."""
        msg = input("Type your message: ")

        # Creating an example Astropy Table
        t = Table()
        t['name'] = ['source 1', 'source 2', 'source 3']
        t['flux'] = [1.2, 2.2, 3.1]

        ret = {
            "message": msg,
            "astropy_table": t,
        }
        return ret,


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="A message publisher and listener for native Python objects and Astropy Tables.")
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
        print("Method result:", msg_object)


if __name__ == "__main__":
    args = parse_args()
    main(args)
