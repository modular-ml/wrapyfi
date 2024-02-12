import argparse

import cv2

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR
from wrapyfi.config.manager import ConfigManager


class ExperimentController(MiddlewareCommunicator):
    WEBCAM_HEIGHT = 240
    WEBCAM_WIDTH = 320
    PEPPER_CAM_HEIGHT = 480
    PEPPER_CAM_WIDTH = 640
    ICUB_CAM_HEIGHT = 240
    ICUB_CAM_WIDTH = 320
    ESR_CAM_HEIGHT = 240
    ESR_CAM_WIDTH = 320

    def __init__(self, **kwargs):
        super(ExperimentController, self).__init__()

    @MiddlewareCommunicator.register(
        "NativeObject",
        "$_mware",
        "ExperimentController",
        "/control_interface/facial_expressions_esr9",
        should_wait=False,
    )
    def listen_facial_expressions_esr9(self, _mware=DEFAULT_COMMUNICATOR):
        return (None,)

    @MiddlewareCommunicator.register(
        "NativeObject",
        "$_mware",
        "ExperimentController",
        "/control_interface/facial_expressions_icub",
        should_wait=False,
    )
    def publish_facial_expressions_icub(self, obj, _mware="yarp"):
        return (obj,)

    @MiddlewareCommunicator.register(
        "NativeObject",
        "$_mware",
        "ExperimentController",
        "/control_interface/facial_expressions_pepper",
        carrier="tcp",
        should_wait=False,
    )
    def publish_facial_expressions_pepper(self, obj, _mware="ros"):
        return (obj,)

    @MiddlewareCommunicator.register(
        "Image",
        "$_mware",
        "ExperimentController",
        "$_topic",
        width="$_width",
        height="$_height",
        rgb=True,
        fp=False,
        should_wait=False,
        jpg=True,
    )
    def listen_image_webcam(
        self,
        _mware=DEFAULT_COMMUNICATOR,
        _width=WEBCAM_WIDTH,
        _height=WEBCAM_HEIGHT,
        _topic="/control_interface/image_webcam",
    ):  # dynamic topic according to corresponding API
        return (None,)

    @MiddlewareCommunicator.register(
        "Image",
        "$_mware",
        "ExperimentController",
        "$_topic",
        width="$_width",
        height="$_height",
        rgb=True,
        fp=False,
        should_wait=False,
    )
    def listen_image_pepper_cam(
        self,
        _mware="ros",
        _width=PEPPER_CAM_WIDTH,
        _height=PEPPER_CAM_HEIGHT,
        _topic="/pepper/camera/front/camera/image_raw",
    ):
        return (None,)

    @MiddlewareCommunicator.register(
        "Image",
        "$_mware",
        "ExperimentController",
        "$_topic",
        width="$_width",
        height="$_height",
        rgb=True,
        fp=False,
        should_wait=False,
    )
    def listen_image_icub_cam(
        self,
        _mware="yarp",
        _width=ICUB_CAM_WIDTH,
        _height=ICUB_CAM_HEIGHT,
        _topic="/icub/cam/right",
    ):
        return (None,)

    @MiddlewareCommunicator.register(
        "Image",
        "$_mware",
        "ExperimentController",
        "/control_interface/image_webcam",
        width="$_width",
        height="$_height",
        rgb=True,
        fp=False,
        should_wait=False,
    )
    def forward_image_webcam(
        self,
        img,
        _mware=DEFAULT_COMMUNICATOR,
        _width=WEBCAM_WIDTH,
        _height=WEBCAM_HEIGHT,
    ):
        return (img,)

    @MiddlewareCommunicator.register(
        "Image",
        "$_mware",
        "ExperimentController",
        "/control_interface/image_pepper_cam",
        width="$_width",
        height="$_height",
        rgb=True,
        fp=False,
        jpg=True,
        should_wait=False,
    )
    def forward_image_pepper_cam(
        self,
        img,
        _mware=DEFAULT_COMMUNICATOR,
        _width=PEPPER_CAM_WIDTH,
        _height=PEPPER_CAM_HEIGHT,
    ):
        return (img,)

    @MiddlewareCommunicator.register(
        "Image",
        "$_mware",
        "ExperimentController",
        "/control_interface/image_icub_cam",
        width="$_width",
        height="$_height",
        rgb=True,
        fp=False,
        jpg=True,
        should_wait=False,
    )
    def forward_image_icub_cam(
        self,
        img,
        _mware=DEFAULT_COMMUNICATOR,
        _width=ICUB_CAM_WIDTH,
        _height=ICUB_CAM_HEIGHT,
    ):
        # convert to bgr
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img,)

    @MiddlewareCommunicator.register(
        "Image",
        "$_mware",
        "ExperimentController",
        "/control_interface/image_esr9",
        width="$_width",
        height="$_height",
        rgb=True,
        fp=False,
        should_wait=False,
        jpg=True,
    )
    def publish_esr9_cam(
        self,
        img,
        _mware=DEFAULT_COMMUNICATOR,
        _width=ESR_CAM_WIDTH,
        _height=ESR_CAM_HEIGHT,
    ):
        # modify size of image
        if img is not None:
            img = cv2.resize(img, (self.ESR_CAM_WIDTH, self.ESR_CAM_HEIGHT))
        return (img,)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrapyfi_cfg",
        help="File to load Wrapyfi configs for running instance. "
        "Choose one of the configs available in "
        "./wrapyfi_configs/affective_signaling_multirobot "
        "for each running instance. All configs with a prefix of COMP must "
        "have corresponding instances running. OPT prefixed configs execute "
        "scripts on a machine connected to either robot (Pepper/iCub) or both, "
        "and at least one must run in addition to COMP.",
        type=str,
    )
    parser.add_argument(
        "--cam_source",
        help="The camera input source being either from a "
        "webcam, Pepper, or iCub. Note that this must be similar for all running "
        "instances even when the two robots are set to display the expressions.",
        type=str,
        default="webcam",
        choices=["webcam", "pepper", "icub"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.wrapyfi_cfg:
        print("Wrapyfi config loading from ", args.wrapyfi_cfg)
        ConfigManager(args.wrapyfi_cfg)

    ec = ExperimentController()
    image_cam = None
    while True:
        if args.cam_source == "webcam":
            (image_webcam,) = ec.listen_image_webcam()
            (image_webcam_linked,) = ec.forward_image_webcam(image_webcam)
            if image_webcam is None:
                image_webcam = image_webcam_linked
            image_cam = image_webcam
            if image_webcam is not None:
                cv2.imshow("Webcam image", image_webcam)
                cv2.waitKey(1)
        if args.cam_source == "pepper":
            (image_pepper_cam,) = ec.listen_image_pepper_cam()
            (image_pepper_cam_linked,) = ec.forward_image_pepper_cam(image_pepper_cam)
            if image_pepper_cam is None:
                image_pepper_cam = image_pepper_cam_linked
            image_cam = image_pepper_cam
            if image_pepper_cam is not None:
                cv2.imshow("Pepper image", image_pepper_cam)
                cv2.waitKey(1)
        if args.cam_source == "icub":
            (image_icub_cam,) = ec.listen_image_icub_cam()
            (image_icub_cam_linked,) = ec.forward_image_icub_cam(image_icub_cam)
            if image_icub_cam is None:
                image_icub_cam = image_icub_cam_linked
            image_cam = image_icub_cam
            if image_icub_cam_linked is not None:
                cv2.imshow("iCub image", image_icub_cam_linked)
                cv2.waitKey(1)

        (image_esr,) = ec.publish_esr9_cam(image_cam)
        (facial_expression,) = ec.listen_facial_expressions_esr9()
        if facial_expression is not None:
            ec.publish_facial_expressions_icub(facial_expression)
            ec.publish_facial_expressions_pepper(facial_expression)
