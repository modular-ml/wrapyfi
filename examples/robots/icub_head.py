"""
iCub Head Controller and Camera Viewer using Wrapyfi for Communication

This script demonstrates the capability to control the iCub robot's head and view its camera feed using
the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern,
allowing for message publishing and listening functionalities between processes or machines.

Demonstrations:
    - Using Image messages for camera feed transmission.
    - Running publishers and listeners concurrently with the yarp.RFModule.
    - Using Wrapyfi for creating a port listener only.

Requirements:
    - Wrapyfi: Middleware communication wrapper (Refer to the Wrapyfi documentation for installation instructions)
    - YARP, ROS, ROS2, ZeroMQ (Refer to the Wrapyfi documentation for installation instructions)
    - iCub Robot and Simulator: Ensure the robot and its simulator are installed and configured.
        When running in simulation mode, the `iCub_SIM` must be running in a standalone terminal
        (Refer to the Wrapyfi documentation for installation instructions)
    - NumPy: Used for creating arrays (Installed with Wrapyfi)
    - SciPy: For applying smoothing filters to the facial expressions (Refer to https://www.scipy.org/install.html for installation instructions)
    - Pexpect: To control the facial expressions using RPC

    Install using pip:
    ``pip install scipy pexpect``

Run:
    # For the list of keyboard controls, refer to the comments in Keyboard Controls.

    # Alternative 1: Simulation Mode
        # Ensure that the `iCub_SIM` is running in a standalone terminal.

        # The listener displays images, and coordinates are published without utilizing Wrapyfi's utilities.

        ``python3 icub_head.py --simulation --get_cam_feed --control_head --control_expressions``

    # Alternative 2: Physical Robot
        # The listener displays images, and coordinates are published without utilizing Wrapyfi's utilities.

        ``python3 icub_head.py --get_cam_feed --control_head --control_expressions``

Keyboard Controls:
    - Head Control:
        - Up/Down: Control the head pitch
        - Right/Left: Control the head yaw
        - A/D: Control the head roll (right/left)
        - R: Reset the head to the initial position
        - Esc: Quit the application
    - Eye Control:
        - W/S: Control the eye pitch (up/down)
        - C/Z: Control the eye yaw (right/left)
        - R: Reset the eye to the initial position
        - Esc: Quit the application
    - Facial Expressions:
        - 0: Neutral
        - 1: Happy
        - 2: Sad
        - 3: Surprise
        - 4: Fear
        - 5: Disgust
        - 6: Anger
        - 7: Contempt
        - 8: Cunning
        - 9: Shy
        - Esc: Quit the application
    - Camera Feed:
        - Esc: Quit the application
"""

import os
import time
import argparse
import logging
from collections import deque

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
ICUB_DEFAULT_COMMUNICATOR = os.environ.get("WRAPYFI_DEFAULT_MWARE", ICUB_DEFAULT_COMMUNICATOR)
ICUB_DEFAULT_COMMUNICATOR = os.environ.get("ICUB_DEFAULT_COMMUNICATOR", ICUB_DEFAULT_COMMUNICATOR)
ICUB_DEFAULT_COMMUNICATOR = os.environ.get("ICUB_DEFAULT_MWARE", ICUB_DEFAULT_COMMUNICATOR)

EMOTION_LOOKUP = {
    "Neutral": [("LIGHTS", "neu")],
    "Happy": [("all", "hap")],
    "Sad": [("LIGHTS", "sad")],
    "Surprise": [("LIGHTS", "sur")],
    "Fear": [("raw", "L04"), ("raw", "R04"), ("raw", "M66")],  # change to array
    "Disgust": [("raw", "L01"), ("raw", "R01"), ("raw", "M66")],  # change to array
    "Anger": [("LIGHTS", "ang")],
    "Contempt": [("raw", "L01"), ("raw", "R09"), ("raw", "ME9")],  # change to array
    "Cunning": [("LIGHTS", "cun")],
    "Shy": [("LIGHTS", "shy")],
    "Evil": [("LIGHTS", "evi")]
}


def cartesian_to_spherical(xyz=None, x=None, y=None, z=None, expand_return=None):
    from operator import xor
    import numpy as np

    assert xor((xyz is not None), all((x is not None, y is not None, z is not None)))

    if expand_return is None:
        expand_return = False if xyz is None else True
    if xyz is None:
        xyz = (x, y, z)

    ptr = np.zeros((3,))
    xy = xyz[0] ** 2 + xyz[1] ** 2
    ptr[0] = np.arctan2(xyz[1], xyz[0])
    ptr[1] = np.arctan2(xyz[2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    # ptr[1] = np.arctan2(np.sqrt(xy), xyz[2])  # for elevation angle defined from Z-axis down
    ptr[2] = np.sqrt(xy + xyz[2] ** 2)
    return ptr if not expand_return else {"p": ptr[0], "t": ptr[1], "r": ptr[2]}


def mode_smoothing_filter(time_series, default, alpha=0.22, beta=0.1, window_length=6, min_count=None):
    import scipy.stats
    if min_count is None:
        min_count = window_length // 2
    mode = scipy.stats.mode(time_series[-window_length:])
    return mode.mode[0] if mode.count >= min_count else default


class ICub(MiddlewareCommunicator, yarp.RFModule):
    """
    ICub head controller, facial expression transmitter and camera viewer. Head control can be achieved following two methods:
    1. Using the `control_head_gaze` method, which controls the head gaze in the spherical coordinate system.
    2. Using the `control_gaze_at_plane` method, which controls the head gaze in the cartesian coordinate system.
    Emotions can be controlled using the `update_facial_expressions` method.
    Camera feed can be viewed using the `receive_images` method.
    """

    MWARE = ICUB_DEFAULT_COMMUNICATOR
    CAP_PROP_FRAME_WIDTH = 320
    CAP_PROP_FRAME_HEIGHT = 240
    HEAD_COORDINATES_PORT = "/control_interface/head_coordinates"
    EYE_COORDINATES_PORT = "/control_interface/eye_coordinates"
    GAZE_PLANE_COORDINATES_PORT = "/control_interface/gaze_plane_coordinates"
    FACIAL_EXPRESSIONS_PORT = "/control_interface/facial_expressions"
    # constants
    FACIAL_EXPRESSIONS_QUEUE_SIZE = 7
    FACIAL_EXPRESSION_SMOOTHING_WINDOW = 6

    def __init__(self, simulation=False, headless=False, get_cam_feed=True,
                 img_width=CAP_PROP_FRAME_WIDTH, img_height=CAP_PROP_FRAME_HEIGHT,
                 control_head=True, set_head_coordinates=True, head_coordinates_port=HEAD_COORDINATES_PORT,
                 control_eyes=True, set_eye_coordinates=True, eye_coordinates_port=EYE_COORDINATES_PORT,
                 ikingaze=False,
                 gaze_plane_coordinates_port=GAZE_PLANE_COORDINATES_PORT,
                 control_expressions=False,
                 set_facial_expressions=True, facial_expressions_port=FACIAL_EXPRESSIONS_PORT,
                 mware=MWARE):
        """
        Initialize the ICub head controller, facial expression transmitter and camera viewer.

        :param simulation: bool: Whether to run the simulation or not
        :param headless: bool: Whether to run the headless mode or not
        :param get_cam_feed: bool: Whether to get (listen) the camera feed or not
        :param img_width: int: Width of the image
        :param img_height: int: Height of the image
        :param control_head: bool: Whether to control the head
        :param set_head_coordinates: bool: Whether to set (publish) the head coordinates
        :param head_coordinates_port: str: Port to receive the head coordinates for controlling the head
        :param control_eyes: bool: Whether to control the eyes
        :param set_eye_coordinates: bool: Whether to set (publish) the eye coordinates
        :param eye_coordinates_port: str: Port to receive the eye coordinates for controlling the eyes
        :param ikingaze: bool: Whether to use the iKinGazeCtrl
        :param gaze_plane_coordinates_port: str: Port to receive the gaze plane coordinates for controlling the head/eyes
        :param control_expressions: bool: Whether to control the facial expressions
        :param set_facial_expressions: bool: Whether to set (publish) the facial expressions
        :param facial_expressions_port: str: Port to receive the facial expressions for controlling the facial expressions
        :param mware: str: Middleware to use
        """

        self.__name__ = "iCubController"
        MiddlewareCommunicator.__init__(self)
        yarp.RFModule.__init__(self)

        self.MWARE = mware
        self.FACIAL_EXPRESSIONS_PORT = facial_expressions_port
        self.GAZE_PLANE_COORDINATES_PORT = gaze_plane_coordinates_port
        self.HEAD_COORDINATES_PORT = head_coordinates_port
        self.EYE_COORDINATES_PORT = eye_coordinates_port

        self.headless = headless
        self.ikingaze = ikingaze

        # prepare a property object
        props = yarp.Property()
        props.put("device", "remote_controlboard")
        props.put("local", "/client/head")

        if simulation:
            props.put("remote", "/icubSim/head")
            self.cam_props = {"cam_world_port": "/icubSim/cam",
                              "cam_left_port": "/icubSim/cam/left",
                              "cam_right_port": "/icubSim/cam/right"}
            emotion_cmd = f"yarp rpc /icubSim/face/emotions/in"
        else:
            props.put("remote", "/icub/head")
            self.cam_props = {"cam_world_port": "/icub/cam/left",
                              "cam_left_port": "/icub/cam/left",
                              "cam_right_port": "/icub/cam/right"}
            emotion_cmd = f"yarp rpc /icub/face/emotions/in"

        if img_width is not None:
            self.img_width = img_width
            self.CAP_PROP_FRAME_WIDTH = img_width
            self.cam_props["img_width"] = img_width

        if img_height is not None:
            self.img_height = img_height
            self.CAP_PROP_FRAME_HEIGHT = img_height
            self.cam_props["img_height"] = img_height

        if control_expressions:
            if HAVE_PEXPECT:
                # control emotional expressions using RPC
                self.client = pexpect.spawn(emotion_cmd)
            else:
                logging.error("pexpect must be installed to control the emotion interface")
                self.activate_communication(self.update_facial_expressions, "disable")

            self.last_expression = ["", ""]  # (emotion part on the robot's face , emotional expression category)
            self.expressions_queue = deque(maxlen=self.FACIAL_EXPRESSIONS_QUEUE_SIZE)
        else:
            self.activate_communication(self.update_facial_expressions, "disable")

        self._curr_eyes = [0, 0, 0]
        self._curr_head = [0, 0, 0]

        if control_head or control_eyes:
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
                # self.update_head_gaze_speed(head=0.8)
                # self.update_eye_gaze_speed(eye=0.5)
                self.activate_communication(self.control_head_gaze, "disable")
                self.activate_communication(self.control_eye_gaze, "disable")
                self.activate_communication(self._control_head_eye_gaze, "disable")

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

                if not control_head:
                    self.activate_communication(self.control_head_gaze, "disable")
                    self.activate_communication(self.update_head_gaze_speed, "disable")
                if not control_eyes:
                    self.activate_communication(self.control_eye_gaze, "disable")
                    self.activate_communication(self.update_eye_gaze_speed, "disable")

                self.init_pos_head = yarp.Vector(self._num_jnts, self._encs.data())
                self.init_pos_eyes = yarp.Vector(self._num_jnts, self._encs.data())
                self.init_pos = yarp.Vector(self._num_jnts, self._encs.data())

                # set movement speed
                # self.update_head_gaze_speed(pitch=10.0, roll=10.0, yaw=20.0)
                # self.update_eye_gaze_speed(pitch=10.0, yaw=10.0, vergence=20.0)

        else:
            self.activate_communication(self.reset_gaze, "disable")
            self.activate_communication(self.update_head_gaze_speed, "disable")
            self.activate_communication(self.control_head_gaze, "disable")
            self.activate_communication(self.update_eye_gaze_speed, "disable")
            self.activate_communication(self.control_eye_gaze, "disable")
            self.activate_communication(self._control_head_eye_gaze, "disable")
            self.activate_communication(self.wait_for_gaze, "disable")
            self.activate_communication(self.control_gaze_at_plane, "disable")

        if get_cam_feed:
            # control the listening properties from within the app
            self.activate_communication(self.receive_images, "listen")
        if facial_expressions_port:
            if set_facial_expressions:
                self.activate_communication(self.acquire_facial_expressions, "publish")
            else:
                for _ in range(self.FACIAL_EXPRESSIONS_QUEUE_SIZE):
                    self.update_facial_expressions("hap", part="all", smoothing="mode")
                self.activate_communication(self.acquire_facial_expressions, "listen")
        if head_coordinates_port:
            if set_head_coordinates:
                self.activate_communication(self.acquire_head_coordinates, "publish")
            else:
                self.activate_communication(self.acquire_head_coordinates, "listen")
        if eye_coordinates_port:
            if set_eye_coordinates:
                self.activate_communication(self.acquire_eye_coordinates, "publish")
            else:
                self.activate_communication(self.acquire_eye_coordinates, "listen")
        if gaze_plane_coordinates_port:
            self.activate_communication(self.control_gaze_at_plane, "listen")

        self.build()

    def build(self):
        """
        Updates the default method arguments according to constructor arguments. This method is called by the module constructor.
        It is not necessary to call it manually.
        """
        ICub.acquire_head_coordinates.__defaults__ = (self.HEAD_COORDINATES_PORT, None, self.MWARE)
        ICub.acquire_eye_coordinates.__defaults__ = (self.EYE_COORDINATES_PORT, None, self.MWARE)
        ICub.receive_gaze_plane_coordinates.__defaults__ = (self.GAZE_PLANE_COORDINATES_PORT, self.MWARE)
        ICub.wait_for_gaze.__defaults__ = (True, self.MWARE)
        ICub.reset_gaze.__defaults__ = (self.MWARE,)
        ICub.update_head_gaze_speed.__defaults__ = (10.0, 10.0, 20.0, 0.8, self.MWARE)
        ICub.control_head_gaze.__defaults__ = (0.0, 0.0, 0.0, "xyz", self.MWARE)
        ICub.update_eye_gaze_speed.__defaults__ = (10.0, 10.0, 20.0, 0.5, self.MWARE)
        ICub.control_eye_gaze.__defaults__ = (0.0, 0.0, 0.0, self.MWARE)
        ICub._control_head_eye_gaze.__defaults__ = (self.MWARE,)
        ICub.control_gaze_at_plane.__defaults__ = (0, 0, 0.3, 0.3, True, True, self.MWARE)
        ICub.acquire_facial_expressions.__defaults__ = (self.FACIAL_EXPRESSIONS_PORT, None, self.MWARE)
        ICub.update_facial_expressions.__defaults__ = (None, False, "mode", self.MWARE)
        ICub.receive_images.__defaults__ = (self.CAP_PROP_FRAME_WIDTH, self.CAP_PROP_FRAME_HEIGHT, True)

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "$head_coordinates_port",
                                     should_wait=False)
    def acquire_head_coordinates(self, head_coordinates_port=HEAD_COORDINATES_PORT, cv2_key=None,
                                 _mware=MWARE, **kwargs):
        """
        Acquire head coordinates for controlling the iCub.

        :param head_coordinates_port: str: Port to receive head coordinates
        :param cv2_key: int: Key pressed by the user
        :return: dict: Head orientation coordinates
        """

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
            elif cv2_key == 114:  # R key: reset the orientation
                self._curr_head = [0, 0, 0]
                self.reset_gaze()
                logging.info("resetting the orientation")
            else:
                logging.info(cv2_key)  # else print its value
                return None,

            return {"topic": head_coordinates_port.split("/")[-1],
                    "timestamp": time.time(),
                    "pitch": self._curr_head[0],
                    "roll": self._curr_head[1],
                    "yaw": self._curr_head[2],
                    "order": "zyx"},

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "$eye_coordinates_port",
                                     should_wait=False)
    def acquire_eye_coordinates(self, eye_coordinates_port=EYE_COORDINATES_PORT, cv2_key=None,
                                _mware=MWARE, **kwargs):
        """
        Acquire eye coordinates for controlling the iCub.

        :param eye_coordinates_port: str: Port to receive eye coordinates
        :param cv2_key: int: Key pressed by the user
        :return: dict: Eye oreintation coordinates
        """

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
                self.reset_gaze()
                logging.info("resetting the orientation")
            else:
                logging.info(cv2_key)  # else print its value
                return None,

            return {"topic": eye_coordinates_port.split("/")[-1],
                    "timestamp": time.time(),
                    "pitch": self._curr_eyes[0],
                    "yaw": self._curr_eyes[1],
                    "vergence": self._curr_eyes[2]},

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "$gaze_plane_coordinates_port",
                                     should_wait=False)
    def receive_gaze_plane_coordinates(self, gaze_plane_coordinates_port=GAZE_PLANE_COORDINATES_PORT,
                                       _mware=MWARE, **kwargs):
        """
        Receive gaze plane (normalized x,y) coordinates for controlling the iCub.

        :param gaze_plane_coordinates_port: str: Port to receive gaze plane coordinates
        :return: dict: Gaze plane coordinates
        """
        return None,

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "/icub_controller/logs/wait_for_gaze",
                                     should_wait=False)
    def wait_for_gaze(self, reset=True, _mware=MWARE):
        """
        Wait for the gaze actuation to complete.
        :param reset: bool: Whether to reset the gaze location (centre)
        :param _mware: str: Middleware to use
        :return: dict: Gaze waiting log for a given time step
        """
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

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "/icub_controller/logs/reset_gaze",
                                     should_wait=False)
    def reset_gaze(self, _mware=MWARE):
        """
        Reset the eyes and head to their original position.

        :param _mware: str: Middleware to use
        :return: dict: Gaze reset log for a given time step
        """
        self.wait_for_gaze(reset=True)
        return {"topic": "logging_reset_gaze",
                "timestamp": time.time(),
                "command": f"reset gaze"},

    @MiddlewareCommunicator.register("NativeObject", "$mware",
                                     "ICub", "/icub_controller/logs/head_speed",
                                     should_wait=False)
    def update_head_gaze_speed(self, pitch=10.0, roll=10.0, yaw=20.0, head=0.8, _mware=MWARE, **kwargs):
        """
        Control the iCub head speed.

        :param pitch: float->pitch[deg/s]: Pitch speed
        :param roll: float->roll[deg/s]: Roll speed
        :param yaw: float->yaw[deg/s]: Yaw speed
        :param head: float->speed[0,1]: Neck trajectory speed in normalized units (only when using iKinGazeCtrl)
        :param _mware: str: Middleware to use
        :return: dict: Head orientation speed log for a given time step
        """
        if self.ikingaze:
            self._igaze.setNeckTrajTime(head)

            return {"topic": "logging_head_speed",
                    "timestamp": time.time(),
                    "command": f"head speed set to {head}"},
        else:
            self._ipos.setRefSpeed(0, pitch)
            self._ipos.setRefSpeed(1, roll)
            self._ipos.setRefSpeed(2, yaw)

            return {"topic": "logging_head_speed",
                    "timestamp": time.time(),
                    "command": f"head speed set to {pitch, roll, yaw} (pitch, roll, yaw)"},

    @MiddlewareCommunicator.register("NativeObject", "$mware",
                                     "ICub", "/icub_controller/logs/eye_speed",
                                     should_wait=False)
    def update_eye_gaze_speed(self, pitch=10.0, yaw=10.0, vergence=20.0, eye=0.5, _mware=MWARE, **kwargs):
        """
        Control the iCub eye speed.

        :param pitch: float->pitch[deg/s]: Pitch speed
        :param yaw: float->yaw[deg/s]: Yaw speed
        :param vergence: float->vergence[deg/s]: Speed of vergence shift between the eyes
        :param eye: float->speed[0,1]: Eye trajectory speed in normalized units (only when using iKinGazeCtrl)
        :param _mware: str: Middleware to use
        :return: dict: Eye orientation speed log for a given time step
        """
        if self.ikingaze:
            self._igaze.setEyesTrajTime(eye)

            return {"topic": "logging_eye_speed",
                    "timestamp": time.time(),
                    "command": f"eye speed set to {eye}"},
        else:
            self._ipos.setRefSpeed(3, pitch)
            self._ipos.setRefSpeed(4, yaw)
            self._ipos.setRefSpeed(5, vergence)

            return {"topic": "logging_eye_speed",
                    "timestamp": time.time(),
                    "command": f"eye speed set to {pitch, yaw, vergence} (pitch, yaw, vergence)"},

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "/icub_controller/logs/head_orientation_coordinates",
                                     should_wait=False)
    def control_head_gaze(self, pitch=0.0, roll=0.0, yaw=0.0, order="xyz", _mware=MWARE, **kwargs):
        """
        Control the iCub head relative to previous coordinates following the roll,pitch,yaw convention (order=xyz)
        (initialized at 0 looking straight ahead).

        :param pitch: float->pitch[deg]: Pitch angle
        :param roll: float->roll[deg]: Roll angle
        :param yaw: float->yaw[deg]: Yaw angle
        :param order: str: Euler axis order. Only accepts xyz (roll, pitch, yaw)
        :param _mware: str: Middleware to use
        :return: dict: Head orientation coordinates log for a given time step
        """
        if order != "xyz":
            logging.error("only accepts ratation angles following the order='xyz' convention")
            return None,
        # wait for the action to complete
        # self.wait_for_gaze(reset=False)

        # initialize a new tmp vector identical to encs
        self.init_pos_head = yarp.Vector(self._num_jnts, self._encs.data())

        # head control
        self.init_pos_head.set(0, self.init_pos_head.get(0) + pitch)  # tilt/pitch
        self.init_pos_head.set(1, self.init_pos_head.get(1) + roll)  # swing/roll
        self.init_pos_head.set(2, self.init_pos_head.get(2) + yaw)  # pan/yaw

        # self._ipos.positionMove(self.init_pos_head.data())
        self._curr_head = list((pitch, roll, yaw))

        return {"topic": "logging_head_coordinates",
                "timestamp": time.time(),
                "command": f"head orientation set to {self._curr_head} (pitch, roll, yaw)"},

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "/icub_controller/logs/eye_orientation_coordinates",
                                     should_wait=False)
    def control_eye_gaze(self, pitch=0.0, yaw=0.0, vergence=0.0, _mware=MWARE, **kwargs):
        """
        Control the iCub eyes relative to previous coordinates (initialized at 0 looking straight ahead).

        :param pitch: float->pitch[deg]: Pitch angle
        :param yaw: float->yaw[deg]: Yaw (version) angle
        :param vergence: float->yaw[deg]: Vergence angle between the eyes
        :param _mware: str: Middleware to use
        :return: dict: Eye orientation coordinates log for a given time step
        """
        # wait for the action to complete
        # self.wait_for_gaze(reset=False)

        # initialize a new tmp vector identical to encs
        self.init_pos_eyes = yarp.Vector(self._num_jnts, self._encs.data())

        # eye control
        self.init_pos_eyes.set(3, self.init_pos_eyes.get(3) + pitch)  # eye tilt
        self.init_pos_eyes.set(4, self.init_pos_eyes.get(4) + yaw)  # eye pan/version
        self.init_pos_eyes.set(5, self.init_pos_eyes.get(
            5) + vergence)  # the vergence between the eyes (to align, set to 0)

        # self._ipos.positionMove(self.init_pos_eyes.data())
        self._curr_eyes = list((pitch, yaw, vergence))

        return {"topic": "logging_eye_coordinates",
                "timestamp": time.time(),
                "command": f"eye orientation set to {self._curr_eyes} (pitch, yaw, vergence)"},

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "/icub_controller/logs/head_eye_orientation_coordinates",
                                     should_wait=False)
    def _control_head_eye_gaze(self, _mware=MWARE, **kwargs):
        """
        Issue the movement command

        :param _mware: str: Middleware to use
        :return: dict: Head orientation coordinates log for a given time step
        """

        # initialize a new tmp vector identical to encs
        self.init_pos = yarp.Vector(self._num_jnts, self._encs.data())

        # head + eye control
        self.init_pos.set(0, self.init_pos_head.get(0))  # tilt/pitch
        self.init_pos.set(1, self.init_pos_head.get(1))  # swing/roll
        self.init_pos.set(2, self.init_pos_head.get(2))  # pan/yaw
        self.init_pos.set(3, self.init_pos_eyes.get(3))  # eye tilt
        self.init_pos.set(4, self.init_pos_eyes.get(4))  # eye pan/version
        self.init_pos.set(5, self.init_pos_eyes.get(5))  # the vergence between the eyes (to align, set to 0)
        self._ipos.positionMove(self.init_pos.data())

        return {"topic": "logging_head_eye_coordinates",
                "timestamp": time.time(),
                "command": f"head orientation set to {self._curr_head} (pitch, roll, yaw) and eye orientation to {self._curr_eyes} (pitch, yaw, vergence)"},

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "/icub_controller/logs/gaze_plane_coordinates",
                                     should_wait=False)
    def control_gaze_at_plane(self, x=0.0, y=0.0, limit_x=0.3, limit_y=0.3, control_eyes=True, control_head=True,
                              _mware=MWARE, **kwargs):
        """
        Gaze at specific point in a normalized plane in front of the iCub.

        :param x: float->x[-1,1]: x coordinate in the plane limited to the range of -1 (left) and 1 (right)
        :param y: float->y[-1,1]: y coordinate in the plane limited to the range of -1 (bottom) and 1 (top)
        :param limit_x: float->limit_x[0,1]: x coordinate limit in the plane
        :param limit_y: float->limit_y[0,1]: y coordinate limit in the plane
        :param control_eyes: bool: Whether to control the eyes of the robot directly
        :param control_head: bool: Whether to control the head of the robot directly
        :return: dict: Gaze coordinates log for a given time step
        """
        # wait for the action to complete
        # self.wait_for_gaze(reset=False)

        xy = np.array((x, y)) * np.array((limit_x, limit_y))  # limit viewing region
        ptr = cartesian_to_spherical(x=1, y=xy[0], z=-xy[1], expand_return=False)
        # initialize a new tmp vector identical to encs
        ptr_degrees = (np.rad2deg(ptr[0]), np.rad2deg(ptr[1]))

        if control_eyes and control_head:
            if not self.ikingaze:
                logging.error("set ikingaze=True in order to move eyes and head simultaneously")
                return None,
            self.init_pos_ikin = yarp.Vector(3, self._gaze_encs.data())
            self.init_pos_ikin.set(0, ptr_degrees[0])
            self.init_pos_ikin.set(1, ptr_degrees[1])
            self.init_pos_ikin.set(2, 0.0)
            self._igaze.lookAtAbsAngles(self.init_pos_ikin)

        elif control_head:
            if self.ikingaze:
                logging.error("set ikingaze=False in order to move head only")
                return None,
            self.control_head_gaze(pitch=ptr_degrees[1], roll=0, yaw=ptr_degrees[0])
        elif control_eyes:
            if self.ikingaze:
                logging.error("set ikingaze=False in order to move eyes only")
                return None,
            self.control_eye_gaze(pitch=ptr_degrees[1], yaw=ptr_degrees[0], vergence=0)

        return {"topic": "logging_gaze_plane_coordinates",
                "timestamp": time.time(),
                "command": f"moving gaze toward {ptr_degrees} with head={control_head} and eyes={control_eyes}"},

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "$facial_expressions_port",
                                     should_wait=False)
    def acquire_facial_expressions(self, facial_expressions_port=FACIAL_EXPRESSIONS_PORT, cv2_key=None,
                                   _mware=MWARE, **kwargs):
        """
        Acquire facial expressions from the iCub.

        :param facial_expressions_port: str: Port to acquire facial expressions from
        :param cv2_key: int: Key to press to set the facial expression
        :return: dict: Facial expressions log for a given time step
        """
        emotion = None
        if cv2_key is None:
            # TODO (fabawi): listen to stdin for keypress
            logging.error("controlling expressions in headless mode not yet supported")
            return None,
        else:
            if cv2_key == 27:  # Esc key to exit
                exit(0)
            elif cv2_key == -1:  # normally -1 returned,so don"t print it
                pass
            elif cv2_key == 48:  # 0 key: Neutral emotion
                emotion = "Neutral"
                logging.info("expressing neutrality")
            elif cv2_key == 49:  # 1 key: Happy emotion
                emotion = "Happy"
                logging.info("expressing happiness")
            elif cv2_key == 50:  # 2 key: Sad emotion
                emotion = "Sad"
                logging.info("expressing sadness")
            elif cv2_key == 51:  # 3 key: Surprise emotion
                emotion = "Surprise"
                logging.info("expressing surprise")
            elif cv2_key == 52:  # 4 key: Fear emotion
                emotion = "Fear"
                logging.info("expressing fear")
            elif cv2_key == 53:  # 5 key: Disgust emotion
                emotion = "Disgust"
                logging.info("expressing disgust")
            elif cv2_key == 54:  # 6 key: Anger emotion
                emotion = "Anger"
                logging.info("expressing anger")
            elif cv2_key == 55:  # 7 key: Contempt emotion
                emotion = "Contempt"
                logging.info("expressing contempt")
            elif cv2_key == 56:  # 8 key: Cunning emotion
                emotion = "Cunning"
                logging.info("expressing cunningness")
            elif cv2_key == 57:  # 9 key: Shy emotion
                emotion = "Shy"
                logging.info("expressing shyness")
            else:
                logging.info(cv2_key)  # else print its value
                return None,
            return {"topic": facial_expressions_port.split("/")[-1],
                    "timestamp": time.time(),
                    "emotion_category": emotion},

    @MiddlewareCommunicator.register("NativeObject", "$_mware",
                                     "ICub", "/icub_controller/logs/facial_expressions",
                                     should_wait=False)
    def update_facial_expressions(self, expression, part=False, smoothing="mode", _mware=MWARE, **kwargs):
        """
        Control facial expressions of the iCub.

        :param expression: str: Expression to be controlled
        :param expression: str or tuple(str->part, str->emotion) or list[str] or list[tuple(str->part, str->emotion)]:
                            Expression/s abbreviation or matching lookup table entry.
                            If a list is provided, the actions are executed in sequence
        :param part: str: Abbreviation describing parts to control (refer to iCub documentation) ( mou, eli, leb, reb, all, raw, LIGHTS)
        :param smoothing: str: Name of smoothing filter to avoid abrupt changes in emotional expressions
        :return: Emotion log for a given time step
        """
        if expression is None:
            return None,

        if isinstance(expression, (list, tuple)):
            expression = expression[-1]

        if smoothing == "mode":
            self.expressions_queue.append(expression)
            transmitted_expression = mode_smoothing_filter(list(self.expressions_queue), default="neu",
                                                           window_length=self.FACIAL_EXPRESSION_SMOOTHING_WINDOW)
        else:
            transmitted_expression = expression

        expressions_lookup = EMOTION_LOOKUP.get(transmitted_expression, transmitted_expression)
        if isinstance(expressions_lookup, str):
            expressions_lookup = [(part if part else "LIGHTS", expressions_lookup)]

        if self.last_expression[0] == (part if part else "LIGHTS") and self.last_expression[
            1] == transmitted_expression:
            expressions_lookup = []

        for (part_lookup, expression_lookup) in expressions_lookup:
            if part_lookup == "LIGHTS":
                self.client.sendline(f"set leb {expression_lookup}")
                self.client.expect(">>")
                self.client.sendline(f"set reb {expression_lookup}")
                self.client.expect(">>")
                self.client.sendline(f"set mou {expression_lookup}")
                self.client.expect(">>")
                logging.info(f"set leb/reb/mou {expression_lookup}")
            else:
                self.client.sendline(f"set {part_lookup} {expression_lookup}")
                self.client.expect(">>")
                logging.info(f"set {part_lookup} {expression_lookup}")

        self.last_expression[0] = part
        self.last_expression[1] = transmitted_expression

        return {"topic": "logging_facial_expressions",
                "timestamp": time.time(),
                "command": f"emotion set to {part} {expression} with smoothing={smoothing}"},

    @MiddlewareCommunicator.register("Image", "yarp", "ICub", "$cam_world_port",
                                     width="$img_width", height="$img_height", rgb="$_rgb")
    @MiddlewareCommunicator.register("Image", "yarp", "ICub", "$cam_left_port",
                                     width="$img_width", height="$img_height", rgb="$_rgb")
    @MiddlewareCommunicator.register("Image", "yarp", "ICub", "$cam_right_port",
                                     width="$img_width", height="$img_height", rgb="$_rgb")
    def receive_images(self, cam_world_port, cam_left_port, cam_right_port,
                       img_width=CAP_PROP_FRAME_WIDTH, img_height=CAP_PROP_FRAME_HEIGHT, _rgb=True):
        """
        Receive images from the iCub.

        :param cam_world_port: str: Port to receive images from the world camera
        :param cam_left_port: str: Port to receive images from the left camera
        :param cam_right_port: str: Port to receive images from the right camera
        :param img_width: int: Width of the image
        :param img_height: int: Height of the image
        :param _rgb: bool: Whether the image is RGB or not
        :return: Images from the iCub
        """
        external_cam, left_cam, right_cam = None, None, None
        return external_cam, left_cam, right_cam

    def getPeriod(self):
        """
        Get the period of the module.
        :return: float: Period of the module
        """
        return 0.01

    def updateModule(self):
        # print(self.getPeriod())
        external_cam, left_cam, right_cam = self.receive_images(**self.cam_props)
        if external_cam is None:
            external_cam = np.zeros((self.img_height, self.img_width, 1), dtype="uint8")
            left_cam = np.zeros((self.img_height, self.img_width, 1), dtype="uint8")
            right_cam = np.zeros((self.img_height, self.img_width, 1), dtype="uint8")
        else:
            external_cam = cv2.cvtColor(external_cam, cv2.COLOR_BGR2RGB)
            left_cam = cv2.cvtColor(left_cam, cv2.COLOR_BGR2RGB)
            right_cam = cv2.cvtColor(right_cam, cv2.COLOR_BGR2RGB)
        if not self.headless:
            cv2.imshow("ICubCam", np.concatenate((left_cam, external_cam, right_cam), axis=1))
            k = cv2.waitKey(30)
        else:
            k = None

        switch_emotion, = self.acquire_facial_expressions(facial_expressions_port=self.FACIAL_EXPRESSIONS_PORT,
                                                          cv2_key=k, _mware=self.MWARE)
        if switch_emotion is not None and isinstance(switch_emotion, dict):
            self.update_facial_expressions(switch_emotion.get("emotion_category", None),
                                           part=switch_emotion.get("part", False), _mware=self.MWARE)

        # move robot head
        move_robot_head, = self.acquire_head_coordinates(head_coordinates_port=self.HEAD_COORDINATES_PORT,
                                                         cv2_key=k, _mware=self.MWARE)
        if move_robot_head is not None and isinstance(move_robot_head, dict):
            robot_head_speed = move_robot_head.get("speed", False)
            if robot_head_speed and isinstance(robot_head_speed, dict):
                self.update_head_gaze_speed(pitch=robot_head_speed.get("pitch", 10.0),
                                            roll=robot_head_speed.get("roll", 10.0),
                                            yaw=robot_head_speed.get("yaw", 20.0), _mware=self.MWARE)
            if move_robot_head.get("reset_gaze", False):
                self.reset_gaze()
            self.control_head_gaze(pitch=move_robot_head.get("pitch", 0.0),
                                   roll=move_robot_head.get("roll", 0.0),
                                   yaw=move_robot_head.get("yaw", 0.0), _mware=self.MWARE)

        # move robot eyes
        move_robot_eyes, = self.acquire_eye_coordinates(eye_coordinates_port=self.EYE_COORDINATES_PORT,
                                                        cv2_key=k, _mware=self.MWARE)
        if move_robot_eyes is not None and isinstance(move_robot_eyes, dict):
            robot_eye_speed = move_robot_eyes.get("speed", False)
            if robot_eye_speed and isinstance(robot_eye_speed, dict):
                self.update_eye_gaze_speed(pitch=robot_eye_speed.get("pitch", 10.0),
                                           yaw=robot_eye_speed.get("yaw", 10.0),
                                           vergence=robot_eye_speed.get("vergence", 20.0), _mware=self.MWARE)
            if move_robot_eyes.get("reset_gaze", False):
                self.reset_gaze()
            self.control_eye_gaze(pitch=move_robot_eyes.get("pitch", 0.0),
                                  yaw=move_robot_eyes.get("yaw", 0.0),
                                  vergence=move_robot_eyes.get("vergence", 0.0), _mware=self.MWARE)

        if move_robot_head is not None or move_robot_eyes is not None:
            self._control_head_eye_gaze()

        move_robot, = self.receive_gaze_plane_coordinates(gaze_plane_coordinates_port=self.GAZE_PLANE_COORDINATES_PORT,
                                                          _mware=self.MWARE)
        if move_robot is not None and isinstance(move_robot, dict):
            robot_eye_speed = move_robot.get("eye_speed", False)
            if robot_eye_speed and isinstance(robot_eye_speed, dict):
                self.update_eye_gaze_speed(**{"pitch": robot_eye_speed.get("pitch", 10.0),
                                              "yaw": robot_eye_speed.get("yaw", 10.0),
                                              "vergence": robot_eye_speed.get("vergence", 20.0), "_mware": self.MWARE}
                if not self.ikingaze else {"eye": robot_eye_speed.get("eye", 0.5), "_mware": self.MWARE})

            robot_head_speed = move_robot.get("head_speed", False)
            if robot_head_speed and isinstance(robot_head_speed, dict):
                self.update_head_gaze_speed(**{"pitch": robot_head_speed.get("pitch", 10.0),
                                               "roll": robot_head_speed.get("roll", 10.0),
                                               "yaw": robot_head_speed.get("yaw", 20.0), "_mware": self.MWARE}
                if not self.ikingaze else {"head": robot_head_speed.get("head", 0.8), "_mware": self.MWARE})

            if move_robot.get("reset_gaze", False):
                self.reset_gaze()
            self.control_gaze_plane_coordinates(x=move_robot.get("x", 0.0), y=move_robot.get("y", 0.0),
                                                limit_x=move_robot.get("limit_x", 0.3),
                                                limit_y=move_robot.get("limit_y", 0.3),
                                                control_head=move_robot.get("control_head",
                                                                            False if not self.ikingaze else True),
                                                control_eyes=move_robot.get("control_eyes", True), _mware=self.MWARE),

        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", action="store_true", help="Run in simulation")
    parser.add_argument("--headless", action="store_true", help="Disable CV2 GUI")
    parser.add_argument("--ikingaze", action="store_true", help="Enable iKinGazeCtrl")
    parser.add_argument("--get_cam_feed", action="store_true", help="Get the camera feeds from the robot")
    parser.add_argument("--control_head", action="store_true", help="Control the head")
    parser.add_argument("--set_head_coordinates", action="store_true",
                        help="Publish head coordinates set using keyboard commands")
    parser.add_argument("--head_coordinates_port", type=str, default="",
                        help="The port (topic) name used for receiving and transmitting head orientation "
                             "Setting the port name without --set_head_coordinates will only receive the coordinates")
    parser.add_argument("--control_eyes", action="store_true", help="Control the eyes")
    parser.add_argument("--set_eye_coordinates", action="store_true",
                        help="Publish eye coordinates set using keyboard commands")
    parser.add_argument("--eye_coordinates_port", type=str, default="",
                        help="The port (topic) name used for receiving and transmitting eye orientation "
                             "Setting the port name without --set_eye_coordinates will only receive the coordinates")
    parser.add_argument("--gaze_plane_coordinates_port", type=str, default="",
                        help="The port (topic) name used for receiving plane coordinates in 2D for robot to look at")
    parser.add_argument("--control_expressions", action="store_true", help="Control the facial expressions")
    parser.add_argument("--set_facial_expressions", action="store_true",
                        help="Publish facial expressions set using keyboard commands")
    parser.add_argument("--facial_expressions_port", type=str, default="",
                        help="The port (topic) name used for receiving and transmitting facial expressions. "
                             "Setting the port name without --set_facial_expressions will only receive the facial expressions")
    parser.add_argument("--mware", type=str, default=ICUB_DEFAULT_COMMUNICATOR,
                        help="The middleware used for communication. "
                             "This can be overriden by providing either of the following environment variables "
                             "{WRAPYFI_DEFAULT_COMMUNICATOR, WRAPYFI_DEFAULT_MWARE, "
                             "ICUB_DEFAULT_COMMUNICATOR, ICUB_DEFAULT_MWARE}. Defaults to 'yarp'",
                        choices=MiddlewareCommunicator.get_communicators())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert not (args.headless and (args.set_facial_expressions or args.set_head_eye_coordinates)), \
        "setters require a CV2 window for capturing keystrokes. Disable --set-... for running in headless mode"
    # TODO (fabawi): add RPC support for controlling the robot and not just facial expressions. Make it optional
    controller = ICub(**vars(args))
    controller.runModule()