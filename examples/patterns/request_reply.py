import argparse
import cv2

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

parser = argparse.ArgumentParser()
parser.add_argument("--request", dest="mode", action="store_const", const="request", default="request", help="Transmit arguments and await reply")
parser.add_argument("--reply", dest="mode", action="store_const", const="reply", default="request", help="Wait for request and return results/reply")
parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                    help="The middleware to use for transmission")
args = parser.parse_args()


class ReqRep(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", args.mware, "ReqRep", "/req_rep/my_message",
                                     carrier="tcp", persistent=True)
    @MiddlewareCommunicator.register("Image", args.mware, "ReqRep", "/req_rep/my_image_message",
                                     carrier="", width="$img_width", height="$img_height", rgb=True, jpg=True,
                                     persistent=True)
    def send_message(self, msg=None, img_width=320, img_height=240, *args, **kwargs):

        obj = {"message": msg,
               "args": args,
               "kwargs": kwargs}

        img = cv2.imread("../../resources/wrapyfi.png")
        img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)
        # adding text to the image and displaying it
        cv2.putText(img, msg,
                    ((img.shape[1] - cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]) // 2,
                     (img.shape[0] + cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return obj, img


req_rep = ReqRep()
req_rep.activate_communication(ReqRep.send_message, mode=args.mode)

counter = 0
while True:
    # We separate the request and reply to show that messages are passed from the requester,
    # but this separation is NOT necessary for the method to work
    if args.mode == "request":
        msg = input("Type your message: ")
        my_message, my_image = req_rep.send_message(msg, counter=counter)
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
    if args.mode == "reply":
        # The send_message() only executes in "reply" mode,
        # meaning, the method is only accessible from this code block
        my_message, my_image = req_rep.send_message()
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
