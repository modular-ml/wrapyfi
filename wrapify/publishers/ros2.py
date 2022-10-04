import logging
import sys
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
import std_msgs.msg
import sensor_msgs.msg

from wrapify.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapify.middlewares.ros2 import ROS2Middleware
from wrapify.encoders import JsonEncoder


class ROS2Publisher(Publisher, Node):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, queue_size=5, **kwargs):
        ROS2Middleware.activate()
        Publisher.__init__(self, name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        Node.__init__(self, name)
        self.queue_size = queue_size

    def await_connection(self, publisher, out_port=None, repeats=None):
        connected = False
        if out_port is None:
            out_port = self.out_port
        logging.info(f"Waiting for topic subscriber: {out_port}")
        if repeats is None:
            if self.should_wait:
                repeats = -1
            else:
                repeats = 1
            while repeats > 0 or repeats <= -1:
                repeats -= 1
                connected = publisher.get_subscription_count() < 1
                if connected:
                    break
                time.sleep(0.02)
        logging.info(f"Topic subscriber connected: {out_port}")
        return connected

    def close(self):
        """
        Close the node
        :return: None
        """
        if hasattr(self, "_publisher"):
            self.destroy_node()

    def __del__(self):
        self.close()


@Publishers.register("NativeObject", "ros2")
class ROS2NativeObjectPublisher(ROS2Publisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, queue_size=5, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, queue_size=queue_size, **kwargs)
        self._publisher = None
        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        self._publisher = self.create_publisher(std_msgs.msg.String, self.out_port, qos_profile=self.queue_size)
        established = self.await_connection(self._publisher, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, obj):
        if not self.established:
            established = self.establish()
            if not established:
                return
            else:
                time.sleep(0.2)
        obj_str = json.dumps(obj, cls=JsonEncoder)
        obj_str_msg = std_msgs.msg.String()
        obj_str_msg.data = obj_str
        self._publisher.publish(obj_str_msg)


@Publishers.register("Image", "ros2")
class ROS2ImagePublisher(ROS2Publisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, width=-1, height=-1, rgb=True, fp=False, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        if self.fp:
            self._encoding = '32FC3' if self.rgb else '32FC1'
            self._type = np.float32
        else:
            self._encoding = 'bgr8' if self.rgb else 'mono8'
            self._type = np.uint8
        self._publisher = None
        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        # self._publisher = rospy.Publisher(self.out_port, sensor_msgs.msg.Image, queue_size=self.queue_size)
        self._publisher = self.create_publisher(sensor_msgs.msg.Image, self.out_port, qos_profile=self.queue_size)
        established = self.await_connection(self._publisher)
        return self.check_establishment(established)

    def publish(self, img):
        if not self.established:
            established = self.establish()
            if not established:
                return
            else:
                time.sleep(0.2)
        if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
            raise ValueError("Incorrect image shape for publisher")
        img = np.require(img, dtype=self._type, requirements='C')
        msg = sensor_msgs.msg.Image()
        msg.header.stamp = self.get_clock.now()
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = self._encoding
        msg.is_bigendian = img.dtype.byteorder == '>' or (img.dtype.byteorder == '=' and sys.byteorder == 'big')
        msg.step = img.strides[0]
        msg.data = img.tobytes()
        self._publisher.publish(msg)


@Publishers.register("AudioChunk", "ros2")
class ROS2AudioChunkPublisher(ROS2Publisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, channels=1, rate=44100, chunk=-1, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self._publisher = None
        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        # self._publisher = rospy.Publisher(self.out_port, sensor_msgs.msg.Image, queue_size=self.queue_size)
        self._publisher = self.create_publisher(sensor_msgs.msg.Image, self.out_port, qos_profile=self.queue_size)
        established = self.await_connection(self._publisher)
        return self.check_establishment(established)

    def publish(self, aud):
        if not self.established:
            established = self.establish()
            if not established:
                return
            else:
                time.sleep(0.2)
        aud, rate = aud
        if rate != self.rate:
            raise ValueError("Incorrect audio rate for publisher")
        if aud is None:
            return
        if 0 < self.chunk != aud.shape[0] or aud.shape[1] != self.channels:
            raise ValueError("Incorrect audio shape for publisher")
        aud = np.require(aud, dtype=np.float32, requirements='C')
        msg = sensor_msgs.msg.Image()
        msg.header.stamp = self.get_clock().now()
        msg.height = aud.shape[0]
        msg.width = aud.shape[1]
        msg.encoding = '32FC1'
        msg.is_bigendian = aud.dtype.byteorder == '>' or (aud.dtype.byteorder == '=' and sys.byteorder == 'big')
        msg.step = aud.strides[0]
        msg.data = aud.tobytes()
        self._publisher.publish(msg)


@Publishers.register("Properties", "ros2")
class ROS2PropertiesPublisher(ROS2Publisher):
    def __init__(self, name, out_port, **kwargs):
        super().__init__(name, out_port, **kwargs)
        raise NotImplementedError
