"""
A message requester and replier for native Python objects, and images using OpenCV.

This script demonstrates the capability to transmit native Python objects and images using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the REQ/REP pattern
allowing message requesting and replying functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject, Image and AudioChunk messages
    - Transmitting a Python object, an image, and audio chunk
    - Applying the REQ/REP pattern with persistence
    - Reading and transmitting an OpenCV image which can be loaded, resized, and displayed on the client and server ends
    - Reading and transmitting a sounddevice (PortAudio with NumPy) audio chunk which can be played back on the client and server ends

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - OpenCV: Used for handling and creating images (installed with Wrapyfi)
    - sounddevice, soundfile: Used for handling audio

    Install using pip:
        ``pip install sounddevice soundfile``

Run:
    # Alternative 1: Image and NativeObject transmission
        # On machine 1 (or process 1): Requester sends a message and awaits a reply (image and native object)

        ``python3 request_reply_example.py --mode request --stream image``

        # On machine 2 (or process 2): Replier waits for a message and sends a reply (image and native object)

        ``python3 request_reply_example.py --mode reply --stream image``

    # Alternative 2: AudioChunk and NativeObject transmission
        # On machine 1 (or process 1): Requester sends a message and awaits a reply (audio chunk and native object)

        ``python3 request_reply_example.py --mode request --stream audio``

        # On machine 2 (or process 2): Replier waits for a message and sends a reply (audio chunk and native object)

        ``python3 request_reply_example.py --mode reply --stream audio``
"""

import argparse
import logging
import time

import sounddevice as sd
import soundfile as sf
import cv2
import numpy as np

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class ReqRep(MiddlewareCommunicator):

    @MiddlewareCommunicator.register(
        "NativeObject", "$mware", "ReqRep", "/req_rep/my_message",
        carrier="tcp", persistent=True
    )
    @MiddlewareCommunicator.register(
        "Image", "$mware", "ReqRep", "/req_rep/my_image_message",
        carrier="", width="$img_width", height="$img_height", rgb=True, jpg=True,
        persistent=True
    )
    def send_img_message(self, msg=None, img_width=320, img_height=240,
                         mware=None, *args, **kwargs):
        """
        Exchange messages with OpenCV images and other native Python objects.
        """
        obj = {"message": msg, "args": args, "kwargs": kwargs}

        # read image from file
        img = cv2.imread("../../resources/wrapyfi.png")
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
        cv2.putText(img, msg,
                    ((img.shape[1] - cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]) // 2,
                     (img.shape[0] + cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return obj, img

    @MiddlewareCommunicator.register(
        "NativeObject", "$mware", "ReqRep", "/req_rep/my_message",
        carrier="tcp", persistent=True
    )
    @MiddlewareCommunicator.register(
        "AudioChunk", "$mware", "ReqRep", "/req_rep/my_audio_message",
        carrier="", rate="$aud_rate", chunk="$aud_chunk", channels="$aud_channels",
        persistent=True
    )
    def send_aud_message(self, msg=None,
                     aud_rate=-1, aud_chunk=-1, aud_channels=2,
                     mware=None, *args, **kwargs):
        """Exchange messages with sounddevice audio chunks and other native Python objects."""
        obj = {"message": msg, "args": args, "kwargs": kwargs}
        # read audio from file
        aud = sf.read("../../resources/sound_test.wav", dtype="float32")
        # aud = (np.mean(aud[0], axis=1), aud[1])
        return obj, aud


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A message requester and replier for native Python objects, images using OpenCV, and sound using PortAudio.")
    parser.add_argument(
        "--mode", type=str, default="request",
        choices={"request", "reply"},
        help="The mode of communication, either 'request' or 'reply'"
    )
    parser.add_argument(
        "--mware", type=str, default=DEFAULT_COMMUNICATOR,
        choices=MiddlewareCommunicator.get_communicators(),
        help="The middleware to use for transmission"
    )

    parser.add_argument(
        "--sound_device", type=int, default=0,
        help="The sound device to use for audio playback"
    )

    parser.add_argument(
        "--list_sound_devices", action="store_true",
        help="List all available sound devices and exit"
    )

    parser.add_argument(
        "--stream", type=str, default="image", choices={"image", "audio"},
        help="The streamed data as either 'image' or 'audio'"
    )
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

    """Main function to initiate ReqRep class and communication."""
    if args.mode == "request" and args.mware == "zeromq":
        print("WE INTENTIONALLY WAIT 5 SECONDS TO ALLOW THE REPLIER ENOUGH TIME TO START UP. ")
        print("THIS IS NEEDED WHEN USING ZEROMQ AS THE COMMUNICATION MIDDLEWARE IF THE SERVER ")
        print("IS SET TO SPAWN A PROXY BROKER (DEFAULT).")
        time.sleep(5)

    req_rep = ReqRep()
    if args.stream == "image":
        req_rep.activate_communication(ReqRep.send_img_message, mode=args.mode)
        req_rep.activate_communication(ReqRep.send_aud_message, mode="disable")
    elif args.stream == "audio":
        req_rep.activate_communication(ReqRep.send_aud_message, mode=args.mode)
        req_rep.activate_communication(ReqRep.send_img_message, mode="disable")
    counter = 0

    while True:
        # We separate the request and reply to show that messages are passed from the requester,
        # but this separation is NOT necessary for the method to work
        if args.mode == "request":
            msg = input("Type your message: ")
            my_message, my_image = req_rep.send_img_message(msg, counter=counter, mware=args.mware)
            my_message2, my_aud, = req_rep.send_aud_message(msg, counter=counter, mware=args.mware)
            my_message = my_message2 if my_message2 is not None else my_message
            counter += 1
            if my_message is not None:
                print("Request: counter:", counter)
                print("Request: received reply:", my_message)
            if my_image is not None:
                cv2.imshow("Received image", my_image)
                while True:
                    k = cv2.waitKey(1) & 0xFF
                    if not (cv2.getWindowProperty("Received image", cv2.WND_PROP_VISIBLE)):
                        break

                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
                cv2.destroyAllWindows()
            if my_aud is not None:
                print("Request: received audio:", my_aud)
                while True:
                    played = sound_play(my_aud, blocking=True, device=args.sound_device)
                    if played:
                        break
                    else:
                        time.sleep(3)
        if args.mode == "reply":
            # The send_message() only executes in "reply" mode,
            # meaning, the method is only accessible from this code block
            my_message, my_image = req_rep.send_img_message(mware=args.mware)
            my_message2, my_aud, = req_rep.send_aud_message(mware=args.mware)
            my_message = my_message2 if my_message2 is not None else my_message
            if my_message is not None:
                print("Reply: received reply:", my_message)
            if my_image is not None:
                cv2.imshow("Image", my_image)
                while True:
                    k = cv2.waitKey(1) & 0xFF
                    if not (cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE)):
                        break

                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
                cv2.destroyAllWindows()
            if my_aud is not None:
                while True:
                    played = sound_play(my_aud, blocking=True, device=args.sound_device)
                    if played:
                        break
                    else:
                        time.sleep(3)


if __name__ == "__main__":
    args = parse_args()
    main(args)