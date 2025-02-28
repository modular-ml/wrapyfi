"""
A video streaming example using the Video message type with configurable buffering and encoding,
featuring separate threads for network communication and playback.

Run:
    # On publisher machine:
    python3 cam_video_stream.py --mode publish --mware websocket --img_source 0 --buffer_length 30

    # On listener machine:
    python3 cam_video_stream.py --mode listen --mware websocket
"""

import logging
import argparse
import cv2
import numpy as np
import threading
import time
from collections import deque
from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class VideoCam(MiddlewareCommunicator):
    def __init__(
        self,
        *args,
        img_source=0,
        img_width=640,
        img_height=480,
        fps=30,
        buffer_length=30,
        encoder="pyav",
        codec="h264",
        mware=None,
        **kwargs,
    ):
        super().__init__()
        self.img_source = img_source
        self.img_width = img_width
        self.img_height = img_height
        self.fps = fps
        self.buffer_length = buffer_length
        self.encoder = encoder
        self.codec = codec
        self.mware = mware

        self.vid_cap = None

        # Threading and buffer setup
        self.buffer = deque(maxlen=2 * fps)  # Buffer up to 2 seconds of frames
        self.buffer_lock = threading.Lock()
        self.running = threading.Event()
        self.running.set()

    @MiddlewareCommunicator.register(
        "Video",
        "$mware",
        "VideoCam",
        "/video_cam/video_feed",
        carrier="",
        width="$img_width",
        height="$img_height",
        fps="$fps",
        gop="$buffer_length",
        buffer_length="$buffer_length",
        encoder="$encoder",
        codec="$codec",
        queue_size=10,
        should_wait=False
    )
    def collect_cam(self, img_width=640, img_height=480, fps=30,
                    buffer_length=30, encoder="pyav", codec="h264", mware=None):
        if self.vid_cap is None:
            self.vid_cap = cv2.VideoCapture(self.img_source)
            if img_width > 0 and img_height > 0:
                self.vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
                self.vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

        ret, frame = self.vid_cap.read()
        if not ret:
            logging.warning("Failed to capture video frame")
            return None,
        frame = cv2.resize(frame, (img_width, img_height))
        return frame,

    def run_publisher(self):
        """Publish video chunks continuously"""
        self.activate_communication(self.collect_cam, mode="publish")
        while self.running.is_set():
            frame, = self.collect_cam(
                img_width=self.img_width,
                img_height=self.img_height,
                fps=self.fps,
                buffer_length=self.buffer_length,
                encoder=self.encoder,
                codec=self.codec,
                mware=self.mware,
            )

    def run_listener(self):
        """Listen for frames and fill the buffer"""
        self.activate_communication(self.collect_cam, mode="listen")
        while self.running.is_set():
            frame, = self.collect_cam(
                img_width=self.img_width,
                img_height=self.img_height,
                fps=self.fps,
                encoder=self.encoder,
                codec=self.codec,
                mware=self.mware,
            )
            if frame is not None:
                with self.buffer_lock:
                    self.buffer.append(frame)

    def run_playback(self):
        print("Starting video playback")
        cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)
        last_frame_time = time.time()

        while self.running.is_set():
            frame = None
            with self.buffer_lock:
                if self.buffer:
                    frame = self.buffer.popleft()

            if frame is not None:
                # Calculate time since last frame and sleep if necessary
                current_time = time.time()
                elapsed = current_time - last_frame_time
                delay = max(1.0 / self.fps - elapsed, 0.001)  # Ensure minimal sleep to prevent busy-wait
                time.sleep(delay)

                # Display frame
                cv2.imshow("Video Stream", frame)
                last_frame_time = time.time()

            # Process GUI events every iteration (crucial for responsiveness)
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.running.clear()

        cv2.destroyAllWindows()

    def __del__(self):
        if self.vid_cap is not None and self.vid_cap.isOpened():
            self.vid_cap.release()
        self.running.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="Video streaming example")
    parser.add_argument(
        "--mode",
        type=str,
        default="publish",
        choices={"publish", "listen"},
        help="The transmission mode")
    parser.add_argument(
        "--mware",
        type=str,
        default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use")
    parser.add_argument(
        "--img_source",
        type=int,
        default=0,
        help="Camera device index")
    parser.add_argument(
        "--img_width",
        type=int,
        default=640,
        help="Video frame width")
    parser.add_argument(
        "--img_height",
        type=int,
        default=480,
        help="Video frame height")
    parser.add_argument(
        "--buffer_length",
        type=int,
        default=30,
        help="Number of frames per video chunk")
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second")
    parser.add_argument(
        "--codec",
        type=str,
        default="h264",
        help="Video codec (mp4v, mpeg, libx264, h264, etc.)")
    return parser.parse_args()


def main(args):
    video_cam = VideoCam(
        img_source=args.img_source,
        img_width=args.img_width,
        img_height=args.img_height,
        fps=args.fps,
        buffer_length=args.buffer_length,
        codec=args.codec,
        mware=args.mware,
    )

    if args.mode == "publish":
        try:
            video_cam.run_publisher()
        except KeyboardInterrupt:
            video_cam.running.clear()
    elif args.mode == "listen":
        # Listener runs in background thread
        listener_thread = threading.Thread(target=video_cam.run_listener)
        listener_thread.daemon = True
        listener_thread.start()

        # Playback runs in main thread (required for OpenCV GUI)
        try:
            video_cam.run_playback()
        except KeyboardInterrupt:
            video_cam.running.clear()
        listener_thread.join()


if __name__ == "__main__":
    args = parse_args()
    main(args)