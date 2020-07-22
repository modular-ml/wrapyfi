import argparse
import cv2
import numpy as np
import yarp

from wrapify.connect.wrapper import MiddlewareCommunicator

"""
ICub head controller and camera viewer

Here we demonstrate 
1. Using the Image messages
2. Run publishers and listeners in concurrence with the yarp.RFModule
3. Utilizing Wrapify for creating a port listener only


Run:
    # Ensure that the `iCub_SIM` is running in a standalone terminal
    
    # For the list of keyboard controls, go to the comment [# the keyboard commands for controlling the robot]
    
    # Alternative 1 (simulation)
    # Listener shows images and coordinates are published without Wrapify's utilities
    python3 icub_head.py --simulation
    
    # Alternative 2 (physical robot)
    # Listener shows images and coordinates are published without Wrapify's utilities
    python3 icub_head.py
    
"""

yarp.Network.init()


class ICub(MiddlewareCommunicator, yarp.RFModule):
    def __init__(self, simulation=True):
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
        else:
            props.put("remote", "/icub/head")
            self.cam_props = {"port_cam": "/icub/cam",
                              "port_cam_left": "/icub/cam/left",
                              "port_cam_right": "/icub/cam/right"}
        self._curr_eyes = [0, 0, 0]
        self._curr_head = [0, 0, 0]

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

        # control the listening properties from within the app
        self.activate_communication("receive_images", "listen")

    def reset_gaze(self):
        """
        Reset the eyes to their original position
        :return: None
        """
        self._ipos.positionMove(self._encs.data())

        while not self._ipos.checkMotionDone():
            pass

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

    @MiddlewareCommunicator.register("Image", "ICub", "$port_cam",
                                     carrier="", width=320, height=240, rgb=True)
    @MiddlewareCommunicator.register("Image", "ICub", "$port_cam_left",
                                     carrier="", width=320, height=240, rgb=True)
    @MiddlewareCommunicator.register("Image", "ICub", "$port_cam_right",
                                     carrier="", width=320, height=240, rgb=True)
    def receive_images(self, port_cam, port_cam_left, port_cam_right):
        external_cam, left_cam, right_cam = None, None, None
        return external_cam, left_cam, right_cam

    def getPeriod(self):
        return 0.01


    def updateModule(self):
        # print(self.getPeriod())
        external_cam, left_cam, right_cam = self.receive_images(**self.cam_props)
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
            self._curr_head[2] += 1
            print("head yaw left")
        elif k == 81: # Left key
            self._curr_head[2] -= 1
            print("head yaw right")
        elif k == 97: # A key
            self._curr_head[1] += 1
            print("head roll right")
        elif k == 100: # D key
            self._curr_head[1] -= 1
            print("head roll left")
        elif k == 119: # W key
            self._curr_eyes[0] += 1
            print("eye pitch up")
        elif k == 115: # S key
            self._curr_eyes[0] -= 1
            print("eye pitch down")
        elif k == 122:  # Z key
            self._curr_eyes[1] += 1
            print("eye yaw left")
        elif k == 99:  # C key
            self._curr_eyes[1] -= 1
            print("eye yaw right")
        elif k == 114: # R key: reset the pose
            self._curr_eyes = [0,0,0]
            self._curr_head = [0,0,0]
            self.reset_gaze()
        else:
            print(k)  # else print its value

        self.control_gaze(head=self._curr_head, eyes=self._curr_eyes)
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", action="store_true", help="Run in simulation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    controller = ICub(simulation=args.simulation)
    controller.runModule()
