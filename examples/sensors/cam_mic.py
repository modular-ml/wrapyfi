"""
A simple example of using the Wrapper class to publish and listen to audio and video streams.

This script demonstrates the capability to transmit audio and video streams using the 
MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern
allowing message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using the Image and AudioChunk messages
    - Transmitting an audio and video stream
    - Applying the PUB/SUB pattern with persistence
    - Transmitting OpenCV images captured from a camera on the publishing end and displayed on the listener's end
    - Transmitting a sounddevice (PortAudio with NumPy) audio chunk captured from a microphone which can be played back on the listener's end
    - Spawning of multiple processes specifying different functionality for listeners and publishers
    - Using a single return wrapper functionality in conjunction with synchronous callbacks

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS 2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - OpenCV: Used for handling and creating images (installed with Wrapyfi)
    - sounddevice, soundfile: Used for handling audio

    Install using pip:
        ``pip install sounddevice soundfile``

Run:
    # Alternative 1: Separate audio and video publishing
        # On machine 1 (or process 1): The audio stream publishing
        
        ``python3 cam_mic.py --mode publish --stream audio --mic_source 0``
        
        # On machine 2 (or process 2): The video stream publishing
        
        ``python3 cam_mic.py --mode publish --stream video --img_source 0``
        
        # On machine 3 (or process 3): The audio stream listening
        
        ``python3 cam_mic.py --mode listen --stream audio``
        
        # On machine 4 (or process 4): The video stream listening
        
        ``python3 cam_mic.py --mode listen --stream video``
        
    # Alternative 2: Concurrent audio and video publishing
        # On machine 1 (or process 1): The audio/video stream publishing
        
        ``python3 cam_mic.py --mode publish --stream audio video --img_source 0 --mic_source 0``
        
        # On machine 2 (or process 2): The audio/video stream listening
        
        ``python3 cam_mic.py --mode listen --stream audio video``
        
"""

import logging
import argparse

import cv2
import sounddevice as sd
import numpy as np

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class CamMic(MiddlewareCommunicator):
    __registry__ = {}

    def __init__(self, *args, stream=("audio", "video"), mic_source=0,
                 mic_rate=44100, mic_chunk=10000, mic_channels=1, img_source=0,
                 img_width=320, img_height=240, mware=None, **kwargs):
        super(MiddlewareCommunicator, self).__init__()
        self.mic_source = mic_source
        self.mic_rate = mic_rate
        self.mic_chunk = mic_chunk
        self.mic_channels = mic_channels

        self.img_source = img_source
        self.img_width = img_width
        self.img_height = img_height

        self.enable_audio = "audio" in stream
        self.enable_video = self.vid_cap = "video" in stream

        self.mware = mware

    @MiddlewareCommunicator.register("Image", "$mware", "CamMic", "/cam_mic/cam_feed",
                                     carrier="", width="$img_width", height="$img_height", rgb=True, jpg=True,
                                     queue_size=10)
    def collect_cam(self, img_width=320, img_height=240, mware=None):
        """Collect images from the camera."""
        if self.vid_cap is True:
            self.vid_cap = cv2.VideoCapture(self.img_source)
            if img_width > 0 and img_height > 0:
                self.vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
                self.vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
            if not self.vid_cap.isOpened():
                self.vid_cap.release()
                self.vid_cap = False
        if self.vid_cap:
            grabbed, img = self.vid_cap.read()
            if grabbed:
                print("Video frame grabbed")
            else:
                print("Video frame not grabbed")
                img = np.random.randint(256, size=(img_height, img_width, 3), dtype=np.uint8)
        else:
            print("Video capturer not opened")
            img = np.random.randint(256, size=(img_height, img_width, 3), dtype=np.uint8)
        return img,

    @MiddlewareCommunicator.register("AudioChunk", "$mware", "CamMic", "/cam_mic/audio_feed",
                                     carrier="", rate="$mic_rate", chunk="$mic_chunk", channels="$mic_channels")
    def collect_mic(self, aud=None, mic_rate=44100, mic_chunk=int(44100 / 5), mic_channels=1, mware=None):
        """Collect audio from the microphone."""
        aud = aud, mic_rate
        return aud,

    def capture_cam_mic(self):
        """Capture audio and video from the camera and microphone."""
        if self.enable_audio:
            # capture the audio stream from the microphone
            with sd.InputStream(device=self.mic_source, channels=self.mic_channels, callback=self._mic_callback,
                                blocksize=self.mic_chunk, samplerate=self.mic_rate):
                while True:
                    pass
        elif self.enable_video:
            while True:
                self.collect_cam(mware=self.mware)

    def _mic_callback(self, audio, frames, time, status):
        """Callback for the microphone audio stream."""
        if self.enable_video:
            self.collect_cam(img_width=self.img_width, img_height=self.img_height, mware=self.mware)
        self.collect_mic(audio, mic_rate=self.mic_rate, mic_chunk=self.mic_chunk, mic_channels=self.mic_channels, mware=self.mware)
        print(audio.flatten(), audio.min(), audio.mean(), audio.max())

    def __del__(self):
        """Release the video capture device."""
        if not isinstance(self.vid_cap, bool):
            self.vid_cap.release()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A streamer and listener for audio and video streams.")
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"}, help="The transmission mode")
    parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                                                                             help="The middleware to use for transmission")
    parser.add_argument("--stream", nargs="+", default=["video", "audio"],
                        choices={"video", "audio"},
                        help="The streamed sensor data")
    parser.add_argument("--img_source", type=int, default=0, help="The video capture device id (int camera id)")
    parser.add_argument("--img_width", type=int, default=320, help="The image width")
    parser.add_argument("--img_height", type=int, default=240, help="The image height")
    parser.add_argument("--mic_source", type=int, default=None, help="The audio capture device id (int microphone id from python3 -m sounddevice)")
    parser.add_argument("--mic_rate", type=int, default=44100, help="The audio sampling rate")
    parser.add_argument("--mic_channels", type=int, default=1, help="The audio channels")
    parser.add_argument("--mic_chunk", type=int, default=10000, help="The transmitted audio chunk size")
    parser.add_argument("--sound_device", type=int, default=0, help="The sound device to use for audio playback")
    parser.add_argument("--list_sound_devices", action="store_true", help="List all available sound devices and exit")
    return parser.parse_args()


def sound_play(my_aud, blocking=True, device=0):
    """
    Play audio using sounddevice.

    :param my_aud: Tuple[np.ndarray, int]: The audio chunk and sampling rate to play
    :param blocking: bool: Whether to block the execution until the audio is played
    :param device: int: The sound device to use for audio playback
    :return: bool: Whether the audio was played successfully
    """
    try:
        sd.play(*my_aud, blocking=blocking, device=device)
        return True
    except sd.PortAudioError:
        logging.warning("PortAudioError: No device is found or the device is already in use. Will try again in 3 seconds.")
        return False


def main(args):
    if args.list_sound_devices:
        print(sd.query_devices())
        return

    cam_mic = CamMic(stream=args.stream, mic_source=args.mic_source, mware=args.mware)

    if args.mode == "publish":
        cam_mic.activate_communication(CamMic.collect_cam, mode="publish")
        cam_mic.activate_communication(CamMic.collect_mic, mode="publish")
        cam_mic.capture_cam_mic()
    elif args.mode == "listen":
        cam_mic.activate_communication(CamMic.collect_cam, mode="listen")
        cam_mic.activate_communication(CamMic.collect_mic, mode="listen")
        while True:
            if "audio" in args.stream:
                (aud, mic_rate), = cam_mic.collect_mic(mic_rate=args.mic_rate, mic_chunk=args.mic_chunk, mic_channels=args.mic_channels, mware=args.mware)
            else:
                aud = mic_rate = None
            if "video" in args.stream:
                img, = cam_mic.collect_cam(img_source=args.img_source, img_width=args.img_width, img_height=args.img_height, mware=args.mware)
            else:
                img = None
            if img is not None:
                cv2.imshow("Received image", img)
                cv2.waitKey(1)
            if aud is not None:
                sound_play((aud, mic_rate), blocking=False, device=args.sound_device)
                sd.wait(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
