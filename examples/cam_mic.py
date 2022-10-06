import argparse
import cv2
import sounddevice as sd
import numpy as np

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

"""
Camera and Microphone listener + publisher

Here we demonstrate 
1. Using the Image and AudioChunk messages
2. Single return wrapper functionality in conjunction with synchronous callbacks
3. The spawning of multiple processes specifying different functionality for listeners and publishers


Run:
    # Alternative 1
    # On machine 1 (or process 1): The audio stream publishing
    python3 cam_mic.py --mode publish --stream audio --aud-source 0
    # On machine 2 (or process 2): The video stream publishing
    python3 cam_mic.py --mode publish --stream video --img-source 0
    # On machine 3 (or process 3): The audio stream listening
    python3 cam_mic.py --mode listen --stream audio
    # On machine 4 (or process 4): The video stream listening
    python3 cam_mic.py --mode listen --stream video
    
    # Alternative 2 (concurrent audio and video publishing)
    # On machine 1 (or process 1): The audio/video stream publishing
    python3 cam_mic.py --mode publish --stream audio video --img-source 0 --aud-source 0
    # On machine 2 (or process 2): The audio/video stream listening
    python3 cam_mic.py --mode listen --stream audio video
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"}, help="The transmission mode")
    parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                                                                             help="The middleware to use for transmission")
    parser.add_argument("--stream", nargs="+", default=["video", "audio"], choices={"video", "audio"}, help="The streamed sensor data")
    parser.add_argument("--img-source", type=int, default=0, help="The video capture device id (int camera id)")
    parser.add_argument("--img-width", type=int, default=320, help="The image width")
    parser.add_argument("--img-height", type=int, default=240, help="The image height")
    parser.add_argument("--aud-source", type=int, default=None, help="The audio capture device id (int microphone id from python3 -m sounddevice)")
    parser.add_argument("--aud-rate", type=int, default=44100, help="The audio sampling rate")
    parser.add_argument("--aud-channels", type=int, default=1, help="The audio channels")
    parser.add_argument("--aud-chunk", type=int, default=10000, help="The transmitted audio chunk size")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    class CamMic(MiddlewareCommunicator):
        __registry__ = {}

        def __init__(self, *args, stream=("audio", "video"), aud_source=0,
                     aud_rate=44100, aud_chunk=10000, aud_channels=1, img_source=0,
                     img_width=320, img_height=240, **kwargs):
            super(MiddlewareCommunicator, self).__init__()
            self.aud_source = aud_source
            self.aud_rate = aud_rate
            self.aud_chunk = aud_chunk
            self.aud_channels = aud_channels

            self.img_source = img_source
            self.img_width = img_width
            self.img_height = img_height

            self.enable_audio = "audio" in stream
            self.enable_video = self.vid_cap = "video" in stream

        @MiddlewareCommunicator.register("Image", args.mware, "CamMic", "/cam_mic/cam_feed", carrier="", width="$img_width", height="$img_height", rgb=True, queue_size=10)
        def collect_cam(self, img_width=320, img_height=240):
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

        @MiddlewareCommunicator.register("AudioChunk", args.mware, "CamMic", "/cam_mic/audio_feed", carrier="", rate="$aud_rate", chunk="$aud_chunk", channels="$aud_channels")
        def collect_mic(self, aud=None, aud_rate=44100, aud_chunk=int(44100 / 5), aud_channels=1):
            aud = aud, aud_rate
            return aud,

        def capture_cam_mic(self):
            if self.enable_audio:
                # capture the audio stream from the microphone
                with sd.InputStream(device=self.aud_source, channels=self.aud_channels, callback=self._mic_callback, blocksize=self.aud_chunk, samplerate=self.aud_rate):
                    while True:
                        pass
            elif self.enable_video:
                while True:
                    self.collect_cam()

        def _mic_callback(self, audio, frames, time, status):
            if self.enable_video:
                self.collect_cam(img_width=self.img_width, img_height=self.img_height)
            self.collect_mic(audio, aud_rate=self.aud_rate, aud_chunk=self.aud_chunk, aud_channels=self.aud_channels)
            print(audio.flatten(), audio.min(), audio.mean(), audio.max())

        def __del__(self):
            if not isinstance(self.vid_cap, bool):
                self.vid_cap.release()

    cam_mic = CamMic(stream=args.stream, aud_source=args.aud_source)

    if args.mode == "publish":
        cam_mic.activate_communication(CamMic.collect_cam, mode="publish")
        cam_mic.activate_communication(CamMic.collect_mic, mode="publish")
        cam_mic.capture_cam_mic()
    elif args.mode == "listen":
        cam_mic.activate_communication(CamMic.collect_cam, mode="listen")
        cam_mic.activate_communication(CamMic.collect_mic, mode="listen")
        while True:
            if "audio" in args.stream:
                (aud, aud_rate), = cam_mic.collect_mic(aud_rate=args.aud_rate, aud_chunk=args.aud_chunk, aud_channels=args.aud_channels)
            else:
                aud = aud_rate = None
            if "video" in args.stream:
                img, = cam_mic.collect_cam(img_source=args.img_source, img_width=args.img_width, img_height=args.img_height)
            else:
                img = None
            if img is not None:
                cv2.imshow("Received image", img)
                cv2.waitKey(1)
            if aud is not None:
                print(aud.flatten(), aud.min(), aud.mean(), aud.max())
                sd.play(aud.flatten(), samplerate=aud_rate)
                sd.wait(1)
