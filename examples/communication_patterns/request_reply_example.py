"""
A message requester and replier for native Python objects, and images using OpenCV.

This script demonstrates the capability to transmit native Python objects and images using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the REQ/REP pattern
allowing message requesting and replying functionalities between processes or machines.

Demonstrations:
    - Using the NativeObject, Image and AudioChunk messages
    - Transmitting a Python object, an image, and audio chunk
    - Applying the REQ/REP pattern with persistence
    - Transmitting OpenCV image which can be loaded, resized, and displayed on the client and server ends
    - Transmitting a sounddevice (PortAudio with NumPy) audio chunk which can be played back on the client and server ends

Requirements:
    - Wrapyfi: Middleware communication wrapper (refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS2, ZeroMQ (refer to the Wrapyfi documentation for installation instructions)
    - OpenCV: Used for handling and creating images (installed with Wrapyfi)
    - sounddevice, soundfile: Used for handling audio

    Install using pip:
        ``pip install sounddevice soundfile``

Run:
    # On machine 1 (or process 1): Requester sends a message and awaits a reply
        ``python3 request_reply_example.py --mode request``

    # On machine 2 (or process 2): Replier waits for a message and sends a reply
        ``python3 request_reply_example.py --mode reply``
"""

import argparse
import time

import sounddevice as sd
import soundfile as sf
import cv2

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
    @MiddlewareCommunicator.register(
        "AudioChunk", "$mware", "ReqRep", "/req_rep/my_audio_message",
        carrier="", rate="$aud_rate", chunk="$aud_chunk", channels="$aud_channels",
        persistent=True
    )
    def send_message(self, msg=None, img_width=320, img_height=240,
                     aud_rate=-1, aud_chunk=-1, aud_channels=2,
                     mware=None, *args, **kwargs):
        """Exchange messages with OpenCV images and other native Python objects."""
        obj = {"message": msg, "args": args, "kwargs": kwargs}

        # read image from file
        img = cv2.imread("../../resources/wrapyfi.png")
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
        cv2.putText(img, msg,
                    ((img.shape[1] - cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]) // 2,
                     (img.shape[0] + cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # read audio from file
        aud = sf.read("../../resources/sound_test.wav", dtype="float32")
        return obj, img, aud


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A message requester and replier for native Python objects, and images using OpenCV.")
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
    return parser.parse_args()


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
    req_rep.activate_communication(ReqRep.send_message, mode=args.mode)
    counter = 0

    while True:
        # We separate the request and reply to show that messages are passed from the requester,
        # but this separation is NOT necessary for the method to work
        if args.mode == "request":
            msg = input("Type your message: ")
            my_message, my_image, my_aud = req_rep.send_message(msg, counter=counter, mware=args.mware)
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
                    sd.play(*my_aud, blocking=True, device=args.sound_device)
        if args.mode == "reply":
            # The send_message() only executes in "reply" mode,
            # meaning, the method is only accessible from this code block
            my_message, my_image, my_aud = req_rep.send_message(mware=args.mware)
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
                    sd.play(*my_aud, blocking=True, device=args.sound_device)

if __name__ == "__main__":
    args = parse_args()
    main(args)