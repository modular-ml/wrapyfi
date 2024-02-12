"""
Channeling Example using Wrapyfi.

This script demonstrates message channeling through three different middleware (A, B, and C) using
the MiddlewareCommunicator within the Wrapyfi library. It allows message publishing and listening
functionalities between processes or machines.

Demonstrations:
    - Using NativeObject, Image, and AudioChunk messages
    - Transmitting messages through different middlewares (e.g., ROS, YARP, ZeroMQ)
    - Applying the PUB/SUB pattern with channeling

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS 2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - NumPy: Used for creating arrays (installed with Wrapyfi)

    Install using pip:
        ``pip install numpy``

Run:
    # On machine 1 (or process 1): Mode: Publisher sends messages through channels A [ZeroMQ], B [ROS 2], and C [YARP]
        ``python3 channeling_example.py --mode publish --mware_A zeromq --mware_B ros2 --mware_C yarp``

    # On machine 2 (or process 2): Mode: Listener waits for message from channels A [ZeroMQ], B [ROS 2], and C [YARP]
        ``python3 channeling_example.py --mode listen --mware_A zeromq --mware_B ros2 --mware_C yarp``

    # On machine 3 (or process 3): Mode: Listener waits for message from channel C [YARP]
        ``python3 channeling_example.py --mode listen --mware_C yarp``

    # On machine 4 (or process 4) [OPTIONAL]: Mode: Listener waits for message from channel B [ROS 2]
        ``python3 channeling_example.py --mode listen --mware_B ros2``
"""

import argparse
import time

import numpy as np

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class ChannelingCls(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware_A",
        "ChannelingCls",
        "/example/native_A_msg",
        carrier="mcast",
        should_wait=True,
    )
    @MiddlewareCommunicator.register(
        "Image",
        "$mware_B",
        "ChannelingCls",
        "/example/image_B_msg",
        carrier="tcp",
        width="$img_width",
        height="$img_height",
        rgb=True,
        should_wait=False,
    )
    @MiddlewareCommunicator.register(
        "AudioChunk",
        "$mware_C",
        "ChannelingCls",
        "/example/audio_C_msg",
        carrier="tcp",
        rate="$aud_rate",
        chunk="$aud_chunk",
        channels="$aud_channels",
        should_wait=False,
    )
    def read_mulret_mulmware(
        self,
        img_width=200,
        img_height=200,
        aud_rate=44100,
        aud_chunk=8820,
        aud_channels=1,
        mware_A=None,
        mware_B=None,
        mware_C=None,
    ):
        """
        Read and forward messages through channels A, B, and C.
        """
        ros_img = np.random.randint(
            256, size=(img_height, img_width, 3), dtype=np.uint8
        )
        zeromq_aud = (
            np.random.uniform(-1, 1, aud_chunk),
            aud_rate,
        )
        yarp_native = [ros_img, zeromq_aud]
        return yarp_native, ros_img, zeromq_aud


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Channeling Example using Wrapyfi.")
    parser.add_argument(
        "--mode",
        type=str,
        default="publish",
        choices={"publish", "listen"},
        help="The transmission mode",
    )
    parser.add_argument(
        "--mware_A",
        type=str,
        default="none",
        choices=MiddlewareCommunicator.get_communicators().update({"none"}),
        help="The middleware to use for transmission of channel A",
    )
    parser.add_argument(
        "--mware_B",
        type=str,
        default="none",
        choices=MiddlewareCommunicator.get_communicators().update({"none"}),
        help="The middleware to use for transmission of channel B",
    )
    parser.add_argument(
        "--mware_C",
        type=str,
        default="none",
        choices=MiddlewareCommunicator.get_communicators().update({"none"}),
        help="The middleware to use for transmission of channel C",
    )
    return parser.parse_args()


def main(args):
    """
    Main function to initiate ChannelingCls class and communication.
    """
    channeling = ChannelingCls()
    channeling.activate_communication(channeling.read_mulret_mulmware, mode=args.mode)

    while True:
        A_native, B_img, C_aud = channeling.read_mulret_mulmware(
            mware_A=args.mware_A, mware_B=args.mware_B, mware_C=args.mware_C
        )
        if A_native is not None:
            print(f"{args.mware_A}_native::", A_native)
        if B_img is not None:
            print(f"{args.mware_B}_img::", B_img)
        if C_aud is not None:
            print(f"{args.mware_C}_aud::", C_aud)
        time.sleep(0.1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
