import os
import time
import argparse
import logging

import cv2
import numpy as np
import yarp

try:
    import pexpect
    HAVE_PEXPECT = True
except ImportError:
    HAVE_PEXPECT = False
    
from wrapyfi.connect.wrapper import MiddlewareCommunicator

ICUB_DEFAULT_COMMUNICATOR = os.environ.get("WRAPYFI_DEFAULT_COMMUNICATOR", "yarp")
ICUB_DEFAULT_COMMUNICATOR = os.environ.get("ICUB_DEFAULT_MWARE", ICUB_DEFAULT_COMMUNICATOR)

CAMERA_RESOLUTION = (320, 240)  # width, height

"""
ICub head controller and camera viewer

Here we demonstrate 
1. Using the Image messages
2. Run publishers and listeners in concurrence with the yarp.RFModule
3. Utilizing Wrapyfi for creating a port listener only


Run:
    # For the list of keyboard controls, go to the comment [# the keyboard commands for controlling the robot]
    
    # Alternative 1 (simulation)
    # Ensure that the `iCub_SIM` is running in a standalone terminal
    # Listener shows images and coordinates are published without Wrapyfi's utilities
    python3 icub_head.py --simulation --get_cam_feed --control_head --control_expressions
    
    # Alternative 2 (physical robot)
    # Listener shows images and coordinates are published without Wrapyfi's utilities
    python3 icub_head.py --get_cam_feed --control_head --control_expressions
    
"""

EMOTION_LOOKUP = {
    "Neutral": "neu",
    "Happy": "hap",
    "Sad": "sad",
    "Surprise": "sur",
    "Fear": "shy",
    "Disgust": "cun",
    "Anger": "ang",
    "Contempt": "evi"
}


def cartesian_to_spherical(xyz):
    import numpy as np
    ptr = np.zeros((3,))
    xy = xyz[0] ** 2 + xyz[1] ** 2
    ptr[0] = np.arctan2(xyz[1], xyz[0])
    ptr[1] = np.arctan2(xyz[2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    # ptr[1] = np.arctan2(np.sqrt(xy), xyz[2])  # for elevation angle defined from Z-axis down
    ptr[2] = np.sqrt(xy + xyz[2] ** 2)
    return ptr


class ICub(MiddlewareCommunicator, yarp.RFModule):
    def __init__(self, simulation=False, headless=False, get_cam_feed=True,
                 control_head=True,
                 set_head_eye_coordinates=True, head_eye_coordinates_port="/control_interface/head_eye_coordinates",
                 ikingaze=False,
                 gaze_plane_coordinates_port="/control_interface/plane_coordinates",
                 control_expressions=False,
                 set_facial_expressions=True, facial_expressions_port="/emotion_interace/facial_expressions",):
        self.__name__ = "iCubController"
        super(MiddlewareCommunicator, self).__init__()

        self.headless = headless
        self.ikingaze = ikingaze
        self.facial_expressions_port = facial_expressions_port
        self.gaze_plane_coordinates_port = gaze_plane_coordinates_port
        self.head_eye_coordinates_port = head_eye_coordinates_port

        # prepare a property object   
        props = yarp.Property()
        props.put("device", "remote_controlboard")
        props.put("local", "/client/head")

        if simulation:
            props.put("remote", "/icubSim/head")
            self.cam_props = {"port_cam": "/icubSim/cam",
                              "port_cam_left": "/icubSim/cam/left",
                              "port_cam_right": "/icubSim/cam/right"}
            emotion_cmd = f'yarp rpc /icubSim/face/emotions/in'
        else:
            props.put("remote", "/icub/head")
            self.cam_props = {"port_cam": "/icub/cam/left",
                              "port_cam_left": "/icub/cam/left",
                              "port_cam_right": "/icub/cam/right"}
            emotion_cmd = f'yarp rpc /icub/face/emotions/in'

        if control_expressions:
            if HAVE_PEXPECT:
                # control emotional expressions using RPC
                self.client = pexpect.spawn(emotion_cmd)
            else:
                logging.error("pexpect must be installed to control the emotion interface")
                self.activate_communication(ICub.update_facial_expressions, "disable")
        else:
            self.activate_communication(ICub.update_facial_expressions, "disable")
                
        self._curr_eyes = [0, 0, 0]
        self._curr_head = [0, 0, 0]
        
        if control_head:
            if ikingaze:
                self._gaze_encs = yarp.Vector(3, 0.0)
                props_gaze = yarp.Property()
                props_gaze.clear()
                props_gaze.put("device", "gazecontrollerclient")
                props_gaze.put("remote", "/iKinGazeCtrl")
                props_gaze.put("local", "/client/gaze")
                #
                self._gaze_driver = yarp.PolyDriver(props_gaze)

                self._igaze = self._gaze_driver.viewIGazeControl()
                self._igaze.setStabilizationMode(True)

                # set movement speed
                # self._igaze.setNeckTrajTime(0.8)
                # self._igaze.setEyesTrajTime(0.5)
            else:
                # create remote driver
                self._head_driver = yarp.PolyDriver(props)

                # query motor control interfaces
                self._ipos = self._head_driver.viewIPositionControl()
                self._ienc = self._head_driver.viewIEncoders()

                # retrieve number of joints
                self._num_jnts = self._ipos.getAxes()

                logging.info(f"controlling {self._num_jnts} joints")

                # read encoders
                self._encs = yarp.Vector(self._num_jnts)
                self._ienc.getEncoders(self._encs.data())

                # set movement speed
                # self.set_speed_gaze(head_vel=move_robot.get("head_vel", (10.0, 10.0, 20.0)),
                #                        eyes_vel=move_robot.get("eyes_vel", (10.0, 10.0, 20.0)))

        else:
            self.activate_communication(self.reset_gaze, "disable")
            self.activate_communication(self.set_speed_gaze, "disable")
            self.activate_communication(self.control_gaze, "disable")
            self.activate_communication(self.wait_for_gaze, "disable")
            self.activate_communication(self.control_gaze_at_plane, "disable")
            
        if get_cam_feed:
            # control the listening properties from within the app
            self.activate_communication(self.receive_images, "listen")
        if facial_expressions_port:
            if set_facial_expressions:
                self.activate_communication(self.receive_facial_expressions, "publish")
            else:
                self.activate_communication(self.receive_facial_expressions, "listen")
        if head_eye_coordinates_port:
            if set_head_eye_coordinates:
                self.activate_communication(self.receive_head_eye_coordinates, "publish")
            else:
                self.activate_communication(self.receive_head_eye_coordinates, "listen")
        if gaze_plane_coordinates_port:
            self.activate_communication(self.gaze_plane_coordinates_port, "listen")

    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR,
                                     "ICub", "$head_eye_coordinates_port",
                                     should_wait=False)
    def receive_head_eye_coordinates(self, head_eye_coordinates_port="/control_interface/head_eye_coordinates", cv2_key=None):
        if cv2_key is None:
            # TODO (fabawi): listen to stdin for keypress
            logging.error("controlling orientation in headless mode not yet supported")
            return None,
        else:
            if cv2_key == 27:  # Esc key to exit
                exit(0)
            elif cv2_key == -1:  # normally -1 returned,so don't print it
                pass
            # the keyboard commands for controlling the robot
            elif cv2_key == 82:  # Up key
                self._curr_head[0] += 1
                logging.info("head pitch up")
            elif cv2_key == 84:  # Down key
                self._curr_head[0] -= 1
                logging.info("head pitch down")
            elif cv2_key == 83:  # Right key
                self._curr_head[2] -= 1
                logging.info("head yaw left")
            elif cv2_key == 81:  # Left key
                self._curr_head[2] += 1
                logging.info("head yaw right")
            elif cv2_key == 97:  # A key
                self._curr_head[1] -= 1
                logging.info("head roll right")
            elif cv2_key == 100:  # D key
                self._curr_head[1] += 1
                logging.info("head roll left")
            elif cv2_key == 119:  # W key
                self._curr_eyes[0] += 1
                logging.info("eye pitch up")
            elif cv2_key == 115:  # S key
                self._curr_eyes[0] -= 1
                logging.info("eye pitch down")
            elif cv2_key == 122:  # Z key
                self._curr_eyes[1] -= 1
                logging.info("eye yaw left")
            elif cv2_key == 99:  # C key
                self._curr_eyes[1] += 1
                logging.info("eye yaw right")
            elif cv2_key == 114:  # R key: reset the orientation
                self._curr_eyes = [0, 0, 0]
                self._curr_head = [0, 0, 0]
                self.reset_gaze()
                logging.info("resetting the orientation")
            else:
                logging.info(cv2_key)  # else print its value

            return {"topic": "head_eye_coordinates",
                    "timestamp": time.time(),
                    "head": self._curr_head,
                    "eyes": self._curr_eyes},

    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR,
                                     "ICub", "gaze_plane_coordinates_port",
                                     should_wait=False)
    def receive_gaze_plane_coordinates(self, gaze_plane_coordinates_port="/control_interface/plane_coordinates"):
        return None,

    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR,
                                     "ICub", "/icub_controller/logs/wait_for_gaze",
                                     should_wait=False)
    def wait_for_gaze(self, reset=True):
        if self.ikingaze:
            # self._igaze.clearNeckPitch()
            # self._igaze.clearNeckRoll()
            # self._igaze.clearNeckYaw()
            # self._igaze.clearEyes()
            if reset:
                self._igaze.lookAtAbsAngles(self._gaze_encs)
            self._igaze.waitMotionDone(timeout=2.0)
        else:
            if reset:
                self._ipos.positionMove(self._encs.data())
                while not self._ipos.checkMotionDone():
                    pass
        return {"topic": "logging_wait_for_gaze",
                "timestamp": time.time(),
                "command": f"waiting for gaze completed with reset={reset}"},

    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR,
                                     "ICub", "/icub_controller/logs/reset_gaze",
                                     should_wait=False)
    def reset_gaze(self):
        """
        Reset the eyes to their original position
        :return: None
        """
        self.wait_for_gaze(reset=True)
        return {"topic": "logging_reset_gaze",
            "timestamp": time.time(),
            "command": f"reset gaze"},
        
    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR,
                                     "ICub", "/icub_controller/logs/update_head_eye_velocity",
                                     should_wait=False)
    def set_speed_gaze(self, head_vel=(10.0, 10.0, 20.0), eyes_vel=(20.0, 20.0, 20.0)):
        """
        Control the iCub head and eye speeds
        :param head_vel: Head speed (tilt, swing, pan) in deg/sec or int for neck speed (norm) when using iKinGaze
        :param eyes_vel: Eye speed (tilt, pan, divergence) in deg/sec or int for eyes speed (norm) when using iKinGaze
        :return: None
        """
        if self.ikingaze:
            if isinstance(head_vel, tuple):
                head_vel = head_vel[0]
                logging.warning("iKinGaze only supports one speed for the neck, using the first value")
            if isinstance(eyes_vel, tuple):
                eyes_vel = eyes_vel[0]
                logging.warning("iKinGaze only supports one speed for the eyes, using the first value")
            self._igaze.setNeckTrajTime(head_vel)
            self._igaze.setEyesTrajTime(eyes_vel)
        else:
            self._ipos.setRefSpeed(0, head_vel[0])
            self._ipos.setRefSpeed(1, head_vel[1])
            self._ipos.setRefSpeed(2, head_vel[2])
            self._ipos.setRefSpeed(3, eyes_vel[0])
            self._ipos.setRefSpeed(4, eyes_vel[1])
            self._ipos.setRefSpeed(5, eyes_vel[2])
        
        return {"topic": "logging_head_eye_velocity",
                "timestamp": time.time(), 
                "command": f"head set to {head_vel} and eyes set to {eyes_vel}"},
    
    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR,
                                     "ICub", "/icub_controller/logs/update_head_eye_orientation",
                                     should_wait=False)
    def control_gaze(self, head=(0, 0, 0), eyes=(0, 0, 0)):
        """
        Control the iCub head or eyes
        :param head: Head coordinates (tilt, swing, pan) in degrees
        :param eyes: Eye coordinates (tilt, pan, divergence) in degrees
        :return: None
        """
        # wait for the action to complete
        # self.wait_for_gaze(reset=False)

        # initialize a new tmp vector identical to encs
        self.init_pos = yarp.Vector(self._num_jnts, self._encs.data())

        # head control
        self.init_pos.set(0, self.init_pos.get(0) + head[0])  # tilt/pitch
        self.init_pos.set(1, self.init_pos.get(1) + head[1])  # swing/roll
        self.init_pos.set(2, self.init_pos.get(2) + head[2])  # pan/yaw
        # eye control
        self.init_pos.set(3, self.init_pos.get(3) + eyes[0])  # eye tilt
        self.init_pos.set(4, self.init_pos.get(4) + eyes[1])  # eye pan
        self.init_pos.set(5, self.init_pos.get(5) + eyes[2]) # the divergence between the eyes (to align, set to 0)

        self._ipos.positionMove(self.init_pos.data())
        self._curr_head = list(head)
        self._curr_eyes = list(eyes)

        return {"topic": "logging_head_eye_orientation",
                "timestamp": time.time(), 
                "command": f"head set to {head} and eyes set to {eyes}"},

    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR,
                                     "ICub", "/icub_controller/logs/update_head_eye_plane",
                                     should_wait=False)
    def control_gaze_at_plane(self, xy=(0, 0,), limiting_consts_xy=(0.3, 0.3), control_eyes=True, control_head=True):
        """
        Gaze at specific point in a normalized plane in front of the robot

        :param xy: tuple representing the x and y position limited to the range of -1 (bottom left) and 1 (top right)
        :param limiting_consts_xy: tuple representing the x and y position limiting constants
        :param control_eyes: bool indicating whether to control the eyes
        :param control_head: bool indicating whether to control the head
        :return: None
        """
        # wait for the action to complete
        # self.wait_for_gaze(reset=False)

        xy = np.array(xy) * np.array(limiting_consts_xy)  # limit viewing region
        ptr = cartesian_to_spherical((1, xy[0], -xy[1]))
        # initialize a new tmp vector identical to encs
        ptr_degrees = (np.rad2deg(ptr[0]), np.rad2deg(ptr[1]))

        if control_eyes and control_head:
            if not self.ikingaze:
                logging.error("Set ikingaze=True in order to move eyes and head simultaneously")
                return
            self.init_pos_ikin = yarp.Vector(3, self._gaze_encs.data())
            self.init_pos_ikin.set(0, ptr_degrees[0])
            self.init_pos_ikin.set(1, ptr_degrees[1])
            self.init_pos_ikin.set(2, 0.0)
            self._igaze.lookAtAbsAngles(self.init_pos_ikin)

        elif control_head:
            self.control_gaze(head=(ptr_degrees[1], 0, ptr_degrees[0]))
        elif control_eyes:
            self.control_gaze(eyes=(ptr_degrees[1], ptr_degrees[0], 0))

        return {"topic": "logging_head_eye_orientation",
                "timestamp": time.time(),
                "command": f"moving gaze toward {ptr_degrees} with head={control_head} and eyes={control_eyes}"},

    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR,
                                     "ICub", "$facial_expressions_port",
                                     should_wait=False)
    def receive_facial_expressions(self, facial_expressions_port="/emotion_interface/facial_expressions", cv2_key=None):
        emotion = None
        if cv2_key is None:
            # TODO (fabawi): listen to stdin for keypress
            logging.error("controlling expressions in headless mode not yet supported")
            return None,
        else:
            if cv2_key == 27:  # Esc key to exit
                exit(0)
            elif cv2_key == -1:  # normally -1 returned,so don't print it
                pass
            elif cv2_key == 49:  # 1 key: sad emotion
                emotion = "sad"
                logging.info("expressing sadness")
            elif cv2_key == 50:  # 2 key: angry emotion
                emotion = "ang"
                logging.info("expressing anger")
            elif cv2_key == 51:  # 3 key: happy emotion
                emotion = "hap"
                logging.info("expressing happiness")
            elif cv2_key == 52:  # 4 key: neutral emotion
                emotion = "neu"
                logging.info("expressing neutrality")
            elif cv2_key == 53:  # 5 key: surprise emotion
                emotion = "sur"
                logging.info("expressing surprise")
            elif cv2_key == 54:  # 6 key: shy emotion
                emotion = "shy"
                logging.info("expressing shyness")
            elif cv2_key == 55:  # 7 key: evil emotion
                emotion = "evi"
                logging.info("expressing evilness")
            elif cv2_key == 56:  # 8 key: cunning emotion
                emotion = "cun"
                logging.info("expressing cunningness")
            else:
                logging.info(cv2_key)  # else print its value
                return None,
            return {"topic": "facial_expressions",
                    "timestamp": time.time(),
                    "emotion_category": emotion},
    
    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR,
                                     "ICub", "/icub_controller/logs/update_facial_expressions",
                                     should_wait=False)
    def update_facial_expressions(self, expression, part="LIGHTS", smoothing=None):
        """
        Control facial expressions of the iCub
        :param expression: Expression abbreviation
        :param part: Abbreviation describing parts to control (refer to iCub documentation) ( mou, eli, leb, reb, all, LIGHTS)
        :param smoothing: Name of smoothing filter to avoid abrupt changes in emotional expressions
        :return: None
        """
        if expression is None:
            return None,
        if isinstance(expression, (list, tuple)):
            expression = expression[-1]
        expression = EMOTION_LOOKUP.get(expression, expression)

        if part == "LIGHTS":
            self.client.sendline(f"set leb {expression}")
            self.client.expect(">>")
            self.client.sendline(f"set reb {expression}")
            self.client.expect(">>")
            self.client.sendline(f"set mou {expression}")
            self.client.expect(">>")
        else:
            self.client.sendline(f"set {part} {expression}")
            self.client.expect(">>")
            
        return {"topic": "logging_facial_expressions",
                "timestamp": time.time(), 
                "command": f"emotion set to {part} {expression} with smoothing={smoothing}"},

    @MiddlewareCommunicator.register("Image", "yarp", "ICub", "$port_cam",
                                     width="$width", height="$height", rgb="$rgb")
    @MiddlewareCommunicator.register("Image", "yarp", "ICub", "$port_cam_left",
                                     width="$width", height="$height", rgb="$rgb")
    @MiddlewareCommunicator.register("Image", "yarp", "ICub", "$port_cam_right",
                                     width="$width", height="$height", rgb="$rgb")
    def receive_images(self, port_cam, port_cam_left, port_cam_right,
                       width=CAMERA_RESOLUTION[0], height=CAMERA_RESOLUTION[1], rgb=True):
        external_cam, left_cam, right_cam = None, None, None
        return external_cam, left_cam, right_cam

    def getPeriod(self):
        return 0.01

    def updateModule(self):
        # print(self.getPeriod())
        external_cam, left_cam, right_cam = self.receive_images(**self.cam_props)
        if external_cam is None:
            external_cam = np.zeros((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 1), dtype="uint8")
            left_cam = np.zeros((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 1), dtype="uint8")
            right_cam = np.zeros((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 1), dtype="uint8")
        else:
            external_cam = cv2.cvtColor(external_cam, cv2.COLOR_BGR2RGB)
            left_cam = cv2.cvtColor(left_cam, cv2.COLOR_BGR2RGB)
            right_cam = cv2.cvtColor(right_cam, cv2.COLOR_BGR2RGB)
        if not self.headless:
            cv2.imshow("ICubCam", np.concatenate((left_cam, external_cam, right_cam), axis=1))
            k = cv2.waitKey(33)
        else:
            k = None

        switch_emotion, = self.receive_facial_expressions(facial_expressions_port=self.facial_expressions_port, cv2_key=k)
        if switch_emotion is not None and isinstance(switch_emotion, dict):
            self.update_facial_expressions(switch_emotion.get("emotion_category", None), part=switch_emotion.get("part", "LIGHTS"))

        move_robot, = self.receive_head_eye_coordinates(head_eye_coordinates_port=self.head_eye_coordinates_port, cv2_key=k)
        if move_robot is not None and isinstance(move_robot, dict):
            self.set_speed_gaze(head_vel=move_robot.get("head_vel", (10.0, 10.0, 20.0)),
                                eyes_vel=move_robot.get("eyes_vel", (10.0, 10.0, 20.0)))
            if move_robot.get("reset_gaze", False):
                self.reset_gaze()
            self.control_gaze(head=move_robot.get("head", (0, 0, 0)), eyes=move_robot.get("eyes", (0, 0, 0)))

        move_robot, = self.receive_gaze_plane_coordinates(gaze_plane_coordinates_port=self.gaze_plane_coordinates_port)
        if move_robot is not None and isinstance(move_robot, dict):
            self.set_speed_gaze(head_vel=move_robot.get("head_vel", (10.0, 10.0, 20.0) if not self.ikingaze else 0.8),
                                eyes_vel=move_robot.get("eyes_vel", (10.0, 10.0, 20.0) if not self.ikingaze else 0.5))
            if move_robot.get("reset_gaze", False):
                self.reset_gaze()
            self.control_gaze_at_plane(xy=move_robot.get("xy", (0, 0)),
                                        limiting_consts_xy=move_robot.get("limiting_consts_xy", (0.3, 0.3)),
                                        control_head=move_robot.get("control_head", False if not self.ikingaze else True),
                                        control_eyes=move_robot.get("control_eyes", True)),

        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", action="store_true", help="Run in simulation")
    parser.add_argument("--headless", action="store_true", help="Disable CV2 GUI")
    parser.add_argument("--ikingaze", action="store_true", help="Enable iKinGazeCtrl")
    parser.add_argument("--get_cam_feed", action="store_true", help="Get the camera feeds from the robot")
    parser.add_argument("--control_head", action="store_true", help="Control the head and eyes")
    parser.add_argument("--set_head_eye_coordinates", action="store_true",
                        help="Publish head+eye coordinates set using keyboard commands")
    parser.add_argument("--head_eye_coordinates_port", type=str, default="",
                        help="The port (topic) name used for receiving and transmitting head and eye orientation "
                             "Setting the port name without --set_head_eye_coordinates will only receive the coordinates")
    parser.add_argument("--gaze_plane_coordinates_port", type=str, default="",
                        help="The port (topic) name used for receiving plane coordinates in 2D for robot to look at")
    parser.add_argument("--control_expressions", action="store_true", help="Control the facial expressions")
    parser.add_argument("--set_facial_expressions", action="store_true",
                        help="Publish facial expressions set using keyboard commands")
    parser.add_argument("--facial_expressions_port", type=str, default="",
                        help="The port (topic) name used for receiving and transmitting facial expressions. "
                             "Setting the port name without --set_facial_expressions will only receive the facial expressions")

    return parser.parse_args()


if __name__ == "__main__":
    logging.warning("DEPRECATION: The iCub example is transferred to the wrapyfi-interfaces repository and will be removed in version 0.5.0")
    args = parse_args()
    assert not (args.headless and (args.set_facial_expressions or args.set_head_eye_coordinates)), \
        "setters require a CV2 window for capturing keystrokes. Disable --set-... for running in headless mode"
    # TODO (fabawi): add RPC support for controlling the robot and not just facial expressions. Make it optional
    controller = ICub(**vars(args))
    controller.runModule()
