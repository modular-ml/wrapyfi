import functools
import argparse
import time

import cv2
import sounddevice as sd
import numpy as np

from wrapify.connect.wrapper import MiddlewareCommunicator

"""
Video/Image and Audio listener + publisher. This is an extension of cam_mic.py to stream videos and images from 
files as well. 
COMING SOON: Audio file reading support

Here we demonstrate 
1. Using the Image and AudioChunk messages
2. Single return wrapper functionality in conjunction with synchronous callbacks
3. The spawning of multiple processes specifying different functionality for listeners and publishers


Run:
    # Alternative 1
    # On machine 1 (or process 1): The audio stream publishing
    python3 audio_video.py --mode publish --stream audio --aud-source 0
    # On machine 2 (or process 2): The video stream publishing
    python3 audio_video.py --mode publish --stream video --img-source 0
    # On machine 3 (or process 3): The audio stream listening
    python3 audio_video.py --mode listen --stream audio
    # On machine 4 (or process 4): The video stream listening
    python3 audio_video.py --mode listen --stream video
    
    # Alternative 2 (concurrent audio and video publishing)
    # On machine 1 (or process 1): The audio/video stream publishing
    python3 audio_video.py --mode publish --stream audio video --img-source 0 --aud-source 0
    # On machine 2 (or process 2): The audio/video stream listening
    python3 audio_video.py --mode listen --stream audio video
"""


class CamMic(MiddlewareCommunicator):
    def __init__(self, *args, stream=("audio", "video"),
                 aud_source=0, aud_rate=44100, aud_chunk=10000, aud_channels=1,
                 img_source=0, img_width=320, img_height=240, img_fps=30, **kwargs):
        super(MiddlewareCommunicator, self).__init__()
        self.aud_source = aud_source
        self.aud_rate = aud_rate
        self.aud_chunk = aud_chunk
        self.aud_channels = aud_channels
        
        self.img_source = img_source
        self.img_width = img_width
        self.img_height = img_height
        self.img_fps = img_fps

        if "audio" in stream:
            self.enable_audio = True
        else:
            self.enable_audio = False
        if "video" in stream:
            self.vid_cap = cv2.VideoCapture(img_source)
            self.enable_video = True
        else:
            self.enable_video = False

        self.last_img = None

    @MiddlewareCommunicator.register("Image", "CamMic", "$img_port",
                                     carrier="", out_port_connect="$img_port_connect", rgb=True)
    def collect_cam(self, img_port="/cam_mic/cam_feed", img_port_connect="/cam_mic/cam_feed:out"):
        if self.vid_cap.isOpened():
            # wait before you capture
            time.sleep(1/self.img_fps)
            # capture the video stream from the webcam
            grabbed, img = self.vid_cap.read()
            
            if not grabbed:
                print("video not grabbed")
                img = np.random.random((self.img_width, self.img_height, 3)) * 255 if self.last_img is None \
                    else self.last_img
            else:
                self.last_img = img
                print("video grabbed")
        else:
            print("video capturer not opened")
            img = np.random.random((self.img_width, self.img_height, 3)) * 255
        return img,

    @MiddlewareCommunicator.register("AudioChunk", "CamMic", "$aud_port",
                                     carrier="", out_port_connect="$aud_port_connect")
    def collect_mic(self, aud_port="/cam_mic/audio_feed", aud_port_connect="/cam_mic/audio_feed:out", aud=None):
        aud = aud, self.aud_rate
        return aud,

    def capture_cam_mic(self):
        if self.enable_audio:
            # capture the audio stream from the microphone
            with sd.InputStream(device=self.aud_source, channels=self.aud_channels, callback=self.__mic_callback__,
                            blocksize=self.aud_chunk,
                            samplerate=self.aud_rate):
                while True:
                    pass
        elif self.enable_video:
            while True:
                self.collect_cam()

    def __mic_callback__(self, audio, frames, time, status):
        if self.enable_video:
            self.collect_cam(img_width=self.img_width, img_height=self.img_height)
        self.collect_mic(audio, aud_rate=self.aud_rate, aud_chunk=self.aud_chunk, aud_channels=self.aud_channels)

    def __del__(self, exc_type, exc_val, exc_tb):
        self.vid_cap.release()

 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"}, help="The transmission mode")
    parser.add_argument("--stream", nargs="+", default=["video", "audio"], choices={"video", "audio"}, help="The streamed sensor data")
    parser.add_argument("--img-port", type=str, default="/cam_mic/cam_feed", help="The YARP port for publishing/receiving the image")
    parser.add_argument("--img-port-connect", type=str, default="/cam_mic/cam_feed:out", help="The connection name for the output image port")
    parser.add_argument("--img-source", type=str, default="0", help="The video capture device id (int camera id | str video path | str image path)")
    parser.add_argument("--img-width", type=int, default=320, help="The image width")
    parser.add_argument("--img-height", type=int, default=240, help="The image height")
    parser.add_argument("--img-fps", type=int, default=30, help="The video frames per second")
    parser.add_argument("--aud-port", type=str, default="/cam_mic/audio_feed", help="The YARP port for publishing/receiving the audio")
    parser.add_argument("--aud-port-connect", type=str, default="/cam_mic/mic_feed:out", help="The connection name for the output audio port")
    parser.add_argument("--aud-source", type=str, default="0", help="The audio capture device id (int microphone id)")
    parser.add_argument("--aud-rate", type=int, default=44100, help="The audio sampling rate")
    parser.add_argument("--aud-channels", type=int, default=1, help="The audio channels")
    parser.add_argument("--aud-chunk", type=int, default=10000, help="The transmitted audio chunk size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        args.aud_source = int(args.aud_source)
    except:
        args.aud_source = args.aud_source
    try:
        args.img_source = int(args.img_source)
    except:
        args.img_source = args.img_source

    cam_mic = CamMic(stream=args.stream,
                     aud_source=args.aud_source, aud_rate=args.aud_rate, aud_chunk=args.aud_chunk, aud_channels=args.aud_channels,
                     img_source=args.img_source, img_width=args.img_width, img_height=args.img_height, img_fps=args.img_fps)

    # update default params of functions because publisher and listener ports are set before function calls
    cam_mic.collect_cam = functools.partial(cam_mic.collect_cam,
                                            img_port=args.img_port, img_port_connect=args.img_port_connect)
    cam_mic.collect_mic = functools.partial(cam_mic.collect_mic,
                                            aud_port=args.aud_port, aud_port_connect=args.aud_port_connect)

    if args.mode == "publish":
        cam_mic.activate_communication("collect_cam", mode="publish")
        cam_mic.activate_communication("collect_mic", mode="publish")
        cam_mic.capture_cam_mic()
    if args.mode == "listen":
        cam_mic.activate_communication("collect_cam", mode="listen")
        cam_mic.activate_communication("collect_mic", mode="listen")

        while True:
            if "audio" in args.stream:
                aud, = cam_mic.collect_mic()
            else:
                aud = None
            if "video" in args.stream:
                img, = cam_mic.collect_cam()
            else:
                img = None
            if img is not None:
                cv2.imshow("Received Image", img)
                cv2.waitKey(1)
            if aud is not None:
                print(aud)
                sd.play(aud[0].flatten(), samplerate=aud[1])
                sd.wait(1)
