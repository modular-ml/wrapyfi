import argparse

import cv2

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR
from wrapyfi.config.manager import ConfigManager


class ExperimentController(MiddlewareCommunicator):
    WEBCAM_HEIGHT = 240
    WEBCAM_WIDTH = 320
    ICUB_CAM_HEIGHT = 240
    ICUB_CAM_WIDTH = 320
    SIXDREPNET_CAM_HEIGHT = 240
    SIXDREPNET_CAM_WIDTH = 320

    def __init__(self, **kwargs):
        super(ExperimentController, self).__init__()

    @MiddlewareCommunicator.register("NativeObject", "$_sixdrepnet_mware", "ExperimentController",
                                     "/control_interface/orientation_sixdrepnet", should_wait=False)
    @MiddlewareCommunicator.register("NativeObject", "$_waveshareimu_mware", "ExperimentController",
                                     "/control_interface/orientation_waveshareimu", should_wait=False)
    def listen_orientation_sixdrepnet_waveshareimu(self, _sixdrepnet_mware="yarp", _waveshareimu_mware="yarp"):
        return None, None,

    @MiddlewareCommunicator.register("NativeObject", "$_mware", "ExperimentController",
                                     "/control_interface/orientation_icub", should_wait=False)
    def publish_orientation_icub(self, obj, _mware="yarp"):
        return obj,

    @MiddlewareCommunicator.register("NativeObject", "$_mware", "ExperimentController",
                                     "/control_interface/gaze_pupil", should_wait=False)
    def listen_gaze_pupil(self, _mware="zeromq"):
        return None,

    @MiddlewareCommunicator.register("NativeObject", "$_mware", "ExperimentController",
                                     "/control_interface/gaze_icub", should_wait=False)
    def publish_gaze_icub(self, obj, _mware="yarp"):
        return obj,


    @MiddlewareCommunicator.register("Image", "$_mware", "ExperimentController",
                                     "/control_interface/image_webcam",
                                     width="$_width", height="$_height", rgb=True, fp=False, should_wait=False,
                                     jpg=True)
    def listen_image_webcam(self, _mware="ros2", _width=WEBCAM_WIDTH, _height=WEBCAM_HEIGHT):
        return None,

    @MiddlewareCommunicator.register("Image", "$_mware", "ExperimentController",
                                     "/control_interface/image_sixdrepnet",
                                     width="$_width", height="$_height", rgb=True, fp=False, should_wait=False)
    def publish_sixdrepnet_cam(self, img, _mware="ros2", _width=SIXDREPNET_CAM_WIDTH, _height=SIXDREPNET_CAM_HEIGHT):
        # modify size of image
        if img is not None:
            img = cv2.resize(img, (self.SIXDREPNET_CAM_WIDTH, self.SIXDREPNET_CAM_HEIGHT))
        return img,

    def priority_control_sources(self, orientation_sixdrepnet, orientation_imu, control_sources):
        if control_sources[0] == "vision":
            if orientation_sixdrepnet is not None:
                # print("Orientation 6DRepNet: ", orientation_sixdrepnet)
                self.publish_orientation_icub(orientation_sixdrepnet)
            elif orientation_imu is not None and "imu" in control_sources:
                # print("Orientation IMU: ", orientation_imu)
                self.publish_orientation_icub(orientation_imu)
        elif control_sources[0] == "imu":
            if orientation_imu is not None:
                # print("Orientation IMU: ", orientation_imu)
                self.publish_orientation_icub(orientation_imu)
            elif orientation_sixdrepnet is not None and "vision" in control_sources:
                # print("Orientation 6DRepNet: ", orientation_sixdrepnet)
                self.publish_orientation_icub(orientation_sixdrepnet)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapyfi_cfg",
                        help="File to load Wrapyfi configs for running instance. "
                             "Choose one of the configs available in "
                             "./wrapyfi_configs/gaze_mirroring_multisensor. "
                             "for each running instance. All configs with a prefix of COMP must "
                             "have corresponding instances running. OPT prefixed config is optional "
                             "(only when using a vision model) executes the script on a machine connected "
                             "to the camera (or has access to the camera topic).",
                        type=str)
    parser.add_argument("--control_sources",
                        help="Control sources to use for the experiment. The order of sources indicates the priority. "
                             "For example, if vision is the first source, then the vision source will "
                             "be used for control. If vision is not available, then the IMU source will be used. "
                             "If one source is provided, then the experiment will run with that source. ",
                        type=str, default=["vision", "imu"], nargs="+", choices=["vision", "imu"])
    parser.add_argument("--enable_gaze",
                        help="Enable the gaze (eye movement) control. If not enabled, "
                             "then the gaze control will not be used. ",
                        action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.wrapyfi_cfg:
        print("Wrapyfi config loading from ", args.wrapyfi_cfg)
        ConfigManager(args.wrapyfi_cfg)

    ec = ExperimentController()
    image_cam = None
    while True:
        if "vision" in args.control_sources:
            image_webcam, = ec.listen_image_webcam()
            image_webcam_linked, = ec.publish_sixdrepnet_cam(image_cam)
            if image_webcam is None:
                image_webcam = image_webcam_linked
            image_cam = image_webcam
            if image_webcam is not None:
                cv2.imshow("Webcam image", image_webcam)
                cv2.waitKey(1)

        orientation_sixdrepnet, orientation_imu = ec.listen_orientation_sixdrepnet_waveshareimu()
        ec.priority_control_sources(orientation_sixdrepnet, orientation_imu, args.control_sources)

        if args.enable_gaze:
            gaze_pupil, = ec.listen_gaze_pupil()
            if gaze_pupil is not None:
                # print("Gaze Pupil: ", gaze_pupil)
                ec.publish_gaze_icub(gaze_pupil)




