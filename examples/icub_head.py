import time
import argparse
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

class ICub(MiddlewareCommunicator, yarp.RFModule):
    def __init__(self, simulation=False, get_cam_feed=True, 
                 control_head=True, control_expressions=False, 
                 facial_expression_port="/emotion_interace/facial_expression", 
                 head_coordinates_port="/control_interface/head_coordinates"):
        self.__name__ = "iCubController"
        super(MiddlewareCommunicator, self).__init__()

        # prepare a property object   
        props = yarp.Property()
        props.put("device", "remote_controlboard")
        props.put("local", "/client/head")
        if simulation:
            props.put("remote", "/icubSim/head")
            self.cam_props = {"port_cam": "/icubSim/cam",
                              "port_cam_left": "/icubSim/cam/left",
                              "port_cam_right": "/icubSim/cam/right"}
            if control_expressions:
                # control emotional expressions using RPC
                self.client = pexpect.spawn(f'yarp rpc /emotion/in')
        else:
            props.put("remote", "/icub/head")
            self.cam_props = {"port_cam": "/icub/cam/left",
                              "port_cam_left": "/icub/cam/left",
                              "port_cam_right": "/icub/cam/right"}
            if control_expressions:
                if HAVE_PEXPECT:
                     # control emotional expressions using RPC
                    self.client = pexpect.spawn(f'yarp rpc /icub/face/emotions/in')
                else:
                    print("pexpect must be installed to control the emotion interface")
                    self.activate_communication(ICub.update_facial_expression, "disable")
            else:
                self.activate_communication(ICub.update_facial_expression, "disable")
                
        self._curr_eyes = [0, 0, 0]
        self._curr_head = [0, 0, 0]
        
        if control_head:
            # create remote driver
            self._head_driver = yarp.PolyDriver(props)

            # query motor control interfaces
            self._ipos = self._head_driver.viewIPositionControl()
            self._ienc = self._head_driver.viewIEncoders()

            # retrieve number of joints
            self._num_jnts = self._ipos.getAxes()

            print('Controlling', self._num_jnts, 'joints')

            # read encoders
            self._encs = yarp.Vector(self._num_jnts)
            self._ienc.getEncoders(self._encs.data())
        else:
            self.activate_communication(ICub.reset_gaze, "disable")
            self.activate_communication(ICub.set_speed_gaze, "disable")
            self.activate_communication(ICub.control_gaze, "disable")
            
        if get_cam_feed:
            # control the listening properties from within the app
            self.activate_communication(ICub.receive_images, "listen")
        if facial_expressions_port:
            self.activate_communication(ICub.receive_facial_expression, "listen")
        if head_coordinates_port:
            self.activate_communication(ICub.receive_head_coordinates, "listen")
        # set movement speed
        #self.set_speed_gaze(head_vel=move_robot.get("head_vel", (10.0, 10.0, 20.0)),
        #                        eyes_vel=move_robot.get("eyes_vel", (10.0, 10.0, 20.0)))

    
    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR, "ICub", "/control_interface/head_coordinates", should_wait=False)
    def receive_head_coordinates(self):
        return None,
    
    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR, "ICub", "/icub_controller/logs/reset_gaze", should_wait=False)
    def reset_gaze(self):
        """
        Reset the eyes to their original position
        :return: None
        """
        self._ipos.positionMove(self._encs.data())

        while not self._ipos.checkMotionDone():
            pass
         return {"topic": "logging_reset_gaze",
                "timestamp": time.time(), 
                "command": f"reset gaze" },
        
    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR, "ICub", "/icub_controller/logs/update_head_eye_velocity", should_wait=False)
    def set_speed_gaze(self, head_vel=(10.0, 10.0, 20.0), eyes_vel=(20.0, 20.0, 20.0)):
        """
        Control the iCub head and eye speeds
        :param head_vel: Head speed (tilt, swing, pan)
        :param eyes_vel: Eye speed (tilt, pan, divergence)
        :return: None
        """
        self._ipos.setRefSpeed(0, head_vel[0])
        self._ipos.setRefSpeed(1, head_vel[1])
        self._ipos.setRefSpeed(2, head_vel[2])
        self._ipos.setRefSpeed(3, eyes_vel[0])
        self._ipos.setRefSpeed(4, eyes_vel[1])
        self._ipos.setRefSpeed(5, eyes_vel[2])
        
        return {"topic": "logging_head_eye_velocity",
                "timestamp": time.time(), 
                "command": f"head set to {head_vel} and eyes set to {eyes_vel}" },
    
    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR, "ICub", "/icub_controller/logs/update_head_eye_orientation", should_wait=False)
    def control_gaze(self, head=(0, 0, 0), eyes=(0, 0, 0)):
        """
        Control the iCub head and eyes
        :param head: Head coordinates (tilt, swing, pan)
        :param eyes: Eye coordinates (tilt, pan, divergence)
        :return: None
        """
        # wait for the action to complete
        # while not self._ipos.checkMotionDone():
        #     pass

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
                "command": f"head set to {head} and eyes set to {eyes}" },
    
    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR, "ICub", "/emotion_interace/facial_expression", should_wait=False)
    def receive_facial_expression(self):
        return None,
    
    @MiddlewareCommunicator.register("NativeObject", ICUB_DEFAULT_COMMUNICATOR, "ICub", "/icub_controller/logs/update_facial_expression", should_wait=False)
    def update_facial_expression(self, expression, part="LIGHTS", smoothing=None):
        """
        Control facial expressions of the iCub
        :param expression: Expression abbreviation
        :param part: Abbreviation describing parts to control (refer to iCub documentation) ( mou, eli, leb, reb, all, LIGHTS)
        :param smoothing: Name of smoothing filter to avoid abrupt changes in emotional expressions
        :return: None
        """
        expression = EMOTION_LOOKUP.get(expression, expression)
        if part == "LIGHTS":
            self.client.sendline(f"set mou {expression}")
            self.client.expect(">>")
            self.client.sendline(f"set leb {expression}")
            self.client.expect(">>")
            self.client.sendline(f"set reb {expression}")
            self.client.expect(">>")
        else:
            self.client.sendline(f"set {part} {expression}")
            self.client.expect(">>")
            
        return {"topic": "logging_facial_expression",
                "timestamp": time.time(), 
                "command": f"emotion set to {part} {expression} with smoothing={smoothing}"},

    @MiddlewareCommunicator.register("Image", "yarp", "ICub", "$port_cam", carrier="", width=320, height=240, rgb=True)
    @MiddlewareCommunicator.register("Image", "yarp", "ICub", "$port_cam_left", carrier="", width=320, height=240, rgb=True)
    @MiddlewareCommunicator.register("Image", "yarp", "ICub", "$port_cam_right", carrier="", width=320, height=240, rgb=True)
    def receive_images(self, port_cam, port_cam_left, port_cam_right):
        external_cam, left_cam, right_cam = None, None, None
        return external_cam, left_cam, right_cam

    def getPeriod(self):
        return 0.01


    def updateModule(self):
        # print(self.getPeriod())
        
        switch_emotion, = self.receive_facial_expression()
        if switch_emotion is not None and isinstance(switch_emotion, dict):
            self.update_facial_expression(switch_emotion.get("emotion", "hap"))
            return True

        move_robot, = self.receive_head_coordinates()
        if move_robot is not None and isinstance(move_robot, dict):
            self.set_speed_gaze(head_vel=move_robot.get("head_vel", (10.0, 10.0, 20.0)),
                                eyes_vel=move_robot.get("eyes_vel", (10.0, 10.0, 20.0)))
            if move_robot.get("reset_gaze", False):
                self.reset_gaze()
            self.control_gaze(head=move_robot.get("head", (0, 0, 0)), eyes=move_robot.get("eyes", (0, 0, 0)))
            return True

        external_cam, left_cam, right_cam = self.receive_images(**self.cam_props)
        if external_cam is None:
            external_cam = np.zeros((240, 320, 1), dtype = "uint8")
            left_cam = np.zeros((240, 320, 1), dtype = "uint8")
            right_cam = np.zeros((240, 320, 1), dtype = "uint8")
        else:    
            external_cam = cv2.cvtColor(external_cam, cv2.COLOR_BGR2RGB)
            left_cam = cv2.cvtColor(left_cam, cv2.COLOR_BGR2RGB)
            right_cam = cv2.cvtColor(right_cam, cv2.COLOR_BGR2RGB)
        cv2.imshow("ICubCam", np.concatenate((left_cam, external_cam, right_cam), axis=1))
        k = cv2.waitKey(33)
        if k == 27:  # Esc key to exit
            exit(0)
        elif k == -1:  # normally -1 returned,so don't print it
            pass

        # the keyboard commands for controlling the robot
        elif k == 82: # Up key
            self._curr_head[0] += 1
            print("head pitch up")
        elif k == 84: # Down key
            self._curr_head[0] -= 1
            print("head pitch down")
        elif k == 83: # Right key
            self._curr_head[2] -= 1
            print("head yaw left")
        elif k == 81: # Left key
            self._curr_head[2] += 1
            print("head yaw right")
        elif k == 97: # A key
            self._curr_head[1] -= 1
            print("head roll right")
        elif k == 100: # D key
            self._curr_head[1] += 1
            print("head roll left")
        elif k == 119: # W key
            self._curr_eyes[0] += 1
            print("eye pitch up")
        elif k == 115: # S key
            self._curr_eyes[0] -= 1
            print("eye pitch down")
        elif k == 122:  # Z key
            self._curr_eyes[1] -= 1
            print("eye yaw left")
        elif k == 99:  # C key
            self._curr_eyes[1] += 1
            print("eye yaw right")
        elif k == 114: # R key: reset the orientation
            self._curr_eyes = [0,0,0]
            self._curr_head = [0,0,0]
            self.reset_gaze()
        elif k == "110":  # 1 key: sad emotion
            self.update_facial_expression("sad")
        elif k == "UNK":  # 2 key: angry emotion
            self.update_facial_expression("ang")
        elif k == 110:  # 3 key: happy emotion
            self.update_facial_expression("hap")
        elif k == "UNK":  # 4 key: neutral emotion
            self.update_facial_expression("neu")
        elif k == "UNK":  # 5 key: surprise emotion
            self.update_facial_expression("sur")
        elif k == "UNK":  # 6 key: shy emotion
            self.update_facial_expression("shy")
        elif k == 98:  # 7 key: evil emotion
            self.update_facial_expression("evi")
        elif k == "UNK":  # 8 key: cunning emotion
            self.update_facial_expression("cun")
        else:
            print(k)  # else print its value

        self.control_gaze(head=self._curr_head, eyes=self._curr_eyes)
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", action="store_true", help="Run in simulation")
    parser.add_argument("--get_cam_feed", action="store_true", help="Get the camera feeds from the robot")
    parser.add_argument("--control_head", action="store_true", help="Control the head and eyes")
    parser.add_argument("--control_expressions", action="store_true", help="Control the facial expressions")
    parser.add_argument("--facial_expression_port", type=str, default=ICUB_DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(), 
                        help="The port (topic) name used for receiving facial expressions")
    parser.add_argument("--head_coordinates_port", type=str, default=ICUB_DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(), 
                        help="The port (topic) name used for receiving head and eye orientation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # TODO (fabawi): Integrate facial_expression_port and head_coordinates_port
    controller = ICub(**vars(args))
    controller.runModule()
