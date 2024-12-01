"""
This example shows how to use the MiddlewareCommunicator to send and receive webcam images. It can be used to test the
functionality of the middleware using the PUB/SUB pattern with transceive and reemit modes. The example can be run on a
single machine or on multiple machines. In this example, the communication middleware is selected using the ``--mware``
argument. The default middleware is ZeroMQ, but YARP, ROS, and ROS 2 are also supported.

The `send_image` method captures frames from a webcam and publishes them. The webcam source, width, and height can be
configured via command-line arguments. The `apply_effect` method subscribes to the published images, applies a selected
effect (e.g., invert, grayscale, or blur), and republishes the processed image.

WARNING: The reemit script MUST be started before the transceive script when `--should_wait` is passed as an argument.
By default, should_wait is False, which is less efficient as it reenters the method multiple times before receiving an
image. When should wait is True (by passing `--should_wait` to both scripts) it is more efficient but restricts the
script running order to reemit followed to transceive.

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS 2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - OpenCV (for capturing and processing images)

Run:
    # Alternative 1: PUB/SUB mode (transceive and reemit)
        # On machine 1 (or process 1): PUB/SUB mode - Listener receives images, applies effects, and republishes

        ``python3 transceive_reemit_example.py --reemit --mware zeromq --effect invert --img_width -1 --img_height -1``

        # On machine 2 (or process 2): PUB/SUB mode - Publisher captures webcam images and transmits them

        ``python3 transceive_reemit_example.py --transceive --mware zeromq --img_source 0 --img_width -1 --img_height -1``

"""
import argparse

import cv2
import numpy as np

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class CameraEffects(MiddlewareCommunicator):
    def __init__(self, cam_id, img_width, img_height):
        super().__init__()
        self.cap = None if cam_id is None else cv2.VideoCapture(cam_id)
        if img_width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
        if img_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

        if self.cap is not None and not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with ID {cam_id}")

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    @MiddlewareCommunicator.register(
        "Image",
        "$mware",
        "CameraRaw",
        "/camera/raw_image",
        carrier="tcp",
        width="$img_width",
        height="$img_height",
        rgb=True,
        jpg=True,
        queue_size=10,
        should_wait="$should_wait",
        publisher_kwargs={"class_name": "CameraRaw", "out_topic": "/camera/raw_image"},
        listener_kwargs={"class_name": "CameraEffects", "in_topic": "/camera/effect_image"}
    )
    def send_image(self, img_width=320, img_height=240, should_wait=True, mware=None):
        """
        Captures and publishes an image frame from the webcam.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from camera")
            return None,
        return frame,

    @MiddlewareCommunicator.register(
        "Image",
        "$mware",
        "CameraEffects",
        "/camera/effect_image",
        carrier="tcp",
        width="$img_width",
        height="$img_height",
        rgb=True,
        jpg=True,
        queue_size=10,
        should_wait="$should_wait",
        publisher_kwargs={"class_name": "CameraEffects", "out_topic": "/camera/effect_image"},
        listener_kwargs={"class_name": "CameraRaw", "in_topic": "/camera/raw_image"}
    )
    def apply_effect(self, img_from_pub, effect_type="none", img_width=320, img_height=240, should_wait=True, mware=None):
        """
        Applies an effect to the received image frame.
        """
        if img_from_pub is None:
            print("No image received")
            return None,

        try:
            img = np.array(img_from_pub)

            # remove the batch-like dimension if present
            if img.shape[0] == 1:
                img = img[0]

            if effect_type == "invert":
                img_out = cv2.bitwise_not(img)
            elif effect_type == "grayscale":
                img_out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
            elif effect_type == "blur":
                img_out = cv2.GaussianBlur(img, (15, 15), 0)
            else:
                img_out = img  # No effect

            return img_out,

        except Exception as e:
            #print(f"Error processing image: {e}")
            return None,

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
        "--effect",
        type=str,
        default="none",
        choices=["none", "invert", "grayscale", "blur"],
        help="The image effect to apply (none, invert, grayscale, blur)",
    )
    parser.add_argument(
        "--should_wait",
        action='store_true',
        dest='should_wait',
        default=False,
        help="If True, listeners wait for a publisher, and publishers wait for at least one listener",
    )
    parser.add_argument(
        "--mware",
        type=str,
        default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use for transmission",
    )
    parser.add_argument(
        "--img_source",
        type=int,
        default=0,
        help="The video capture device ID (int camera ID)",
    )
    parser.add_argument("--img_width", type=int, default=-1, help="The image width")
    parser.add_argument("--img_height", type=int, default=-1, help="The image height")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()


    if args.mode == "transceive":
        cam_effects = CameraEffects(cam_id=args.img_source, img_width=args.img_width, img_height=args.img_height)
        cam_effects.activate_communication(CameraEffects.send_image, mode="transceive")
        cam_effects.activate_communication(CameraEffects.apply_effect, mode="disable")
    elif args.mode == "reemit":
        cam_effects = CameraEffects(cam_id=None, img_width=args.img_width, img_height=args.img_height)
        cam_effects.activate_communication(CameraEffects.apply_effect, mode="reemit")
        cam_effects.activate_communication(CameraEffects.send_image, mode="disable")

    while True:
        if args.mode == "transceive":
            img_out, = cam_effects.send_image(img_width=args.img_width, img_height=args.img_height, should_wait=args.should_wait, mware=args.mware)
            if img_out is not None:
                cv2.imshow("Captured Image", img_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        elif args.mode == "reemit":
            img_effect, = cam_effects.apply_effect(img_width=args.img_width, img_height=args.img_height, mware=args.mware, should_wait=args.should_wait, effect_type=args.effect)
            if img_effect is not None:
                # print(f"Effect {args.effect} applied successfully.")
                pass