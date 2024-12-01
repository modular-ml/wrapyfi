"""
This example shows how to use the MiddlewareCommunicator to send and receive images. It can be used to test the
functionality of the middleware using the PUB/SUB pattern with transceive and reemit modes. The example can be run on a single
machine or on multiple machines. In this example (as with all other examples), the communication middleware is selected
using the ``--mware`` argument. The default is ZeroMQ, but YARP, ROS, and ROS 2 are also supported.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS 2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)

Run:
    # Alternative 1: PUB/SUB mode (transceive and reemit)
        # On machine 1 (or process 1): PUB/SUB mode - Publisher waits for keyboard input and transmits message

        ``python3 transceive_reemit_example.py --transceive --mware zeromq``

        # On machine 2 (or process 2): PUB/SUB mode - Listener waits for message and prints the received object

        ``python3 transceive_reemit_example.py --reemit --mware zeromq``


"""

### TODO (fabawi): Currently testing with plain messages. Next will apply image transceiving and reemiting
import argparse

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class CameraEffects(MiddlewareCommunicator):
    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware",
        "CameraRaw",
        "/SHOULD_NOT_BE_USED",
        carrier="tcp",
        should_wait=True,
        listener_kwargs={"class_name": "CameraEffects", "in_topic": "/camera/color_invert"},
        publisher_kwargs={"class_name": "CameraRaw", "out_topic": "/camera/raw_image"}
    )
    def send_image(self, arg_from_requester="", mware=None):
        """
        Exchange messages and mirror user input.
        """
        msg = input("Type your message: ")
        obj = {"message": msg, "message_from_requester": arg_from_requester}
        return obj,

    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware",
        "CameraEffects",
        "/SHOULD_NOT_BE_USED",
        carrier="tcp",
        should_wait=True,
        listener_kwargs={"class_name": "CameraRaw", "in_topic": "/camera/raw_image"},
        publisher_kwargs={"class_name": "CameraEffects", "out_topic": "/camera/color_invert"}
    )
    def apply_effect(self,  *img_from_pub, arg_from_requester="", mware=None):
        """
        Apply transform to message.
        """
        msg = input("Type your message: ")
        msg = f"****MSG FROM TRANSCEIVER: {img_from_pub[0]}   #### MSG FROM REEMITER: {msg}"
        obj = {"message": msg, "message_from_requester": arg_from_requester}
        return obj,


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transceive",
        dest="mode",
        action="store_const",
        const="transceive",
        default="transceive",
        help="Transceive mode - publish the method and listen for output instead of just returning published output",
    )
    parser.add_argument(
        "--reemit",
        dest="mode",
        action="store_const",
        const="reemit",
        default="transceive",
        help="Reemit mode - listen for output, apply transformation, and publish transformation",
    )
    parser.add_argument(
        "--mware",
        type=str,
        default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use for transmission",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cam_effects = CameraEffects()
    if args.mode == "transceive":
        cam_effects.activate_communication(CameraEffects.send_image, mode="transceive")
        cam_effects.activate_communication(CameraEffects.apply_effect, mode="disable")
    elif args.mode == "reemit":
        cam_effects.activate_communication(CameraEffects.apply_effect, mode="reemit")
        cam_effects.activate_communication(CameraEffects.send_image, mode="disable")
    while True:
        my_image, = cam_effects.send_image(
            arg_from_requester=f"I got this message from the script running in {args.mode} mode",
            mware=args.mware,
        )
        my_effect, = cam_effects.apply_effect(
            arg_from_requester=f"I got this message from the script running in {args.mode} mode",
            mware=args.mware,
        )
        if my_image is not None:
            print("Method result:", my_image)
        if my_effect is not None:
            print("Method result:", my_effect)
