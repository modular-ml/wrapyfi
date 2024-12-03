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
script running order to reemit followed to transceive. Additionally, `--should_wait` guarantees that all messages are
transmitted and received by the methods. When should_wait is False, the first of the returns to be received will be
processed, while the others could potentially be ignored.

Combinations:
    - transceive: should_wait; reeimit: should_wait => run reemit before transceive. On stopping transceive, reemit has to be restarted
    - transceive: NO WAITING; reeimit: should_wait => run reemit before transceive. On stopping transceive, reemit has to be restarted
    - transceive: should_wait; reeimit: NO WAITING => run reemit before transceive.  No need to restart reemit
    - transceive: NO WAITING; reeimit: NO WAITING => run reemit and transceive in any order. No need to restart reemit


Limitations:
    - Currently does not work with ZeroMQ if should_wait is True (TODO: Fix)
    - Currently does not work with ZeroMQ if no debug publisher exists since it spawns a ZeroMQ publish middleware. Need to fix singleton classes for both (TODO: Fix)
    - Currently does not work with YARP if should_wait is False (TODO: Fix)
    - Currently does not work with Websockets due to race condition with multiple message instances (especially on receiving images) - https://github.com/miguelgrinberg/python-socketio/issues/403

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS 2, Zenoh (refer to the Wrapyfi documentation for installation instructions)
    - OpenCV (for capturing and processing images)

Run:
    # Alternative 1: PUB/SUB mode (transceive and reemit)
        # On machine 1 (or process 1): PUB/SUB mode - Listener receives images, applies effects, and republishes

        ``python3 transceive_reemit_example.py --reemit --mware zenoh --effect invert --img_width -1 --img_height -1 --should_wait``

        # On machine 2 (or process 2): PUB/SUB mode - Publisher captures webcam images and transmits them

        ``python3 transceive_reemit_example.py --transceive --mware zenoh --img_source 0 --img_width -1 --img_height -1 --should_wait``

"""
import argparse
import time
from collections import deque

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

        self.capture_times = deque(maxlen=100)  # to calculate FPS for capturing
        self.frame_timestamps = {}  # stores timestamps of each frame keyed by unique ID
        self.frame_id_counter = 0  # to count frames processed by apply_effect

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def calculate_fps(self, times):
        if len(times) < 2:
            return 0
        return len(times) / (times[-1] - times[0])

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
        multi_threaded=True,
        should_wait="$should_wait",
        publisher_kwargs={"class_name": "CameraRaw", "out_topic": "/camera/raw_image"},
        listener_kwargs={"class_name": "CameraEffects", "in_topic": "/camera/effect_image"}
    )
    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware",
        "CameraEffects",
        "/message/my_message_snd",
        carrier="tcp",
        multi_threaded=True,
        should_wait=False,
        publisher_kwargs={"class_name": "CameraRaw", "out_topic": "/message/my_message_snd"},
        listener_kwargs={"class_name": "CameraEffects", "in_topic": "/message/my_message_rec"}
    )
    def send_image(self, img_width=320, img_height=240, should_wait=True, mware=None):
        """
        Captures and publishes an image frame from the webcam.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from camera")
            return None, None

        # Generate unique ID and store timestamp
        frame_id = self.frame_id_counter
        self.frame_id_counter += 1
        self.frame_timestamps[frame_id] = time.time()

        # Calculate capture FPS
        self.capture_times.append(self.frame_timestamps[frame_id])
        fps = self.calculate_fps(self.capture_times)

        return frame, {"capture_fps": fps, "frame_id": frame_id,
                       "timestamp": self.frame_timestamps[frame_id]}

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
        multi_threaded=True,
        should_wait="$should_wait",
        publisher_kwargs={"class_name": "CameraEffects", "out_topic": "/camera/effect_image"},
        listener_kwargs={"class_name": "CameraRaw", "in_topic": "/camera/raw_image"}
    )
    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware",
        "CameraEffects",
        "/message/my_message_rec",
        carrier="tcp",
        multi_threaded=True,
        should_wait=False,
        publisher_kwargs={"class_name": "CameraEffects", "out_topic": "/message/my_message_rec"},
        listener_kwargs={"class_name": "CameraRaw", "in_topic": "/message/my_message_snd"}
    )
    def apply_effect(self, *data_from_pub, effect_type="none", img_width=320, img_height=240, should_wait=True, mware=None):
        """
        Applies an effect to the received image frame.
        """
        try:
            img_from_pub, msg_from_pub = data_from_pub
        except:
            img_from_pub = None
            msg_from_pub = None

        if img_from_pub is None:
            # print("No image received")
            return None, None

            # Record receive time
        self.capture_times.append(time.time())
        receive_fps = self.calculate_fps(self.capture_times)

        try:
            img = np.array(img_from_pub)

            if effect_type == "invert":
                img_out = cv2.bitwise_not(img)
            elif effect_type == "grayscale":
                img_out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)  # convert back to 3 channels
            elif effect_type == "blur":
                img_out = cv2.GaussianBlur(img, (15, 15), 0)
            else:
                img_out = img  # no effect

            # update the message dictionary
            if msg_from_pub is None:
                msg_from_pub = {}
            if isinstance(msg_from_pub, dict):
                msg_from_pub["received_fps"] = receive_fps

            return img_out, msg_from_pub

        except Exception as e:
            # print(f"Error processing image: {e}")
            return None, None

    @MiddlewareCommunicator.register(
        "NativeObject",
        "$mware",
        "Debug",
        "/COLDSTART",
        carrier="tcp",
        multi_threaded=True,
        should_wait="$should_wait",
        publisher_kwargs={"class_name": "CameraEffects", "out_topic": "/COLDSTART"},
        listener_kwargs={"class_name": "CameraRaw", "in_topic": "/COLDSTART"}
    )
    def debug(self, should_wait=False, mware=None):
        return "hello",

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
        cam_effects.activate_communication(CameraEffects.debug, mode="listen")
        cam_effects.activate_communication(CameraEffects.send_image, mode="transceive")
        cam_effects.activate_communication(CameraEffects.apply_effect, mode="disable")

        last_display_time = time.time()
        display_times = deque(maxlen=100)
        frame_timestamps = {}

    elif args.mode == "reemit":
        cam_effects = CameraEffects(cam_id=None, img_width=args.img_width, img_height=args.img_height)
        cam_effects.activate_communication(CameraEffects.debug, mode="publish")
        cam_effects.activate_communication(CameraEffects.apply_effect, mode="reemit")
        cam_effects.activate_communication(CameraEffects.send_image, mode="disable")

    metrics = {}

    while True:
        testing, = cam_effects.debug(should_wait=args.should_wait, mware=args.mware)
        if testing is not None:
            # print(f"{testing}")
            pass

        if args.mode == "transceive":
            current_display_time = time.time()
            display_times.append(current_display_time)
            if len(display_times) > 1:
                display_fps = len(display_times) / (display_times[-1] - display_times[0])
            else:
                display_fps = 0

            img_out, recent_metrics = cam_effects.send_image(
                img_width=args.img_width,
                img_height=args.img_height,
                should_wait=args.should_wait,
                mware=args.mware
            )

            if recent_metrics is not None:
                metrics = recent_metrics
            frame_id = metrics.get("frame_id")
            timestamp_sent = metrics.get("timestamp")
            if timestamp_sent and timestamp_sent > current_display_time:
                latency = current_display_time - timestamp_sent
            elif timestamp_sent and timestamp_sent < current_display_time:
                latency = timestamp_sent - current_display_time
            else:
                latency = 0
            overlay_text = (
                f"Capture FPS: {metrics.get('capture_fps', 0):.2f}\n"
                f"Received FPS: {metrics.get('received_fps', 0):.2f}\n"
                f"Display FPS: {display_fps:.2f}\n"
                f"Latency: {latency * 1000:.2f} ms"
            )
            if img_out is not None and metrics is not None:
                y_offset = img_out.shape[0] - 10  # Start from the bottom of the image
                for i, line in enumerate(overlay_text.split('\n')):
                    cv2.putText(
                        img_out, line,
                        (10, y_offset - (i * 20)),  # Position the text lines
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),  # White color text
                        1,
                        cv2.LINE_AA
                    )
                cv2.imshow("Captured Image", img_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif img_out is not None:
                cv2.imshow("Captured Image", img_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        elif args.mode == "reemit":
            testing = cam_effects.debug(should_wait=args.should_wait, mware=args.mware)
            img_effect, _ = cam_effects.apply_effect(img_width=args.img_width, img_height=args.img_height, mware=args.mware, should_wait=args.should_wait, effect_type=args.effect)
