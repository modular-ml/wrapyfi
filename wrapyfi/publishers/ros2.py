import logging
import sys
import json
import time
import os
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapyfi.middlewares.ros2 import ROS2Middleware
from wrapyfi.encoders import JsonEncoder


QUEUE_SIZE = int(os.environ.get("WRAPYFI_ROS2_QUEUE_SIZE", 5))
WATCHDOG_POLL_REPEAT = None


class ROS2Publisher(Publisher, Node):

    def __init__(self, name: str, out_port: str, carrier: str = "tcp", should_wait: bool = True,
                 queue_size: int = QUEUE_SIZE, ros2_kwargs: Optional[dict] = None, **kwargs):
        """
        Initialize the publisher

        :param name: str: Name of the publisher
        :param out_port: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS2 currently only supports TCP for pub/sub pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param ros2_kwargs: dict: Additional kwargs for the ROS2 middleware
        :param kwargs: dict: Additional kwargs for the publisher
        """
        if carrier != "tcp":
            logging.warning("ROS2 does not support other carriers than TCP for pub/sub pattern. Using TCP.")
            carrier = "tcp"
        ROS2Middleware.activate(**ros2_kwargs or {})
        Publisher.__init__(self, name, out_port, carrier=carrier, should_wait=should_wait, **kwargs)
        Node.__init__(self, name)

        self.queue_size = queue_size

    def await_connection(self, publisher, out_port: Optional[str] = None, repeats: Optional[int] = None):
        """
        Wait for at least one subscriber to connect to the publisher

        :param publisher: rclpy.publisher.Publisher: Publisher to await connection to
        :param out_port: str: Name of the output topic
        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
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
                connected = publisher.get_subscription_count() > 0
                if connected:
                    break
                time.sleep(0.02)
        logging.info(f"Topic subscriber connected: {out_port}")
        return connected

    def close(self):
        """
        Close the publisher
        """
        if hasattr(self, "_publisher") and self._publisher:
            if self._publisher is not None:
               self.destroy_node()

    def __del__(self):
        self.close()


@Publishers.register("NativeObject", "ros2")
class ROS2NativeObjectPublisher(ROS2Publisher):

    def __init__(self, name, out_port: str, carrier: str = "tcp", should_wait: bool = True,
                 queue_size: int = QUEUE_SIZE, serializer_kwargs: Optional[dict] = None, **kwargs):
        """
        The NativeObject publisher using the ROS2 String message assuming a combination of python native objects
        and numpy arrays as input. Serializes the data (including plugins) using the encoder and sends it as a string

        :param name: str: Name of the publisher
        :param out_port: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS2 currently only supports TCP for pub/sub pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        """
        super().__init__(name, out_port, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)
        self._publisher = None

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        self._publisher = self.create_publisher(std_msgs.msg.String, self.out_port, qos_profile=self.queue_size)
        established = self.await_connection(self._publisher, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, obj):
        """
        Publish the object to the middleware

        :param obj: object: Object to publish
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)
        obj_str = json.dumps(obj, cls=self._plugin_encoder, **self._plugin_kwargs,
                             serializer_kwrags=self._serializer_kwargs)
        obj_str_msg = std_msgs.msg.String()
        obj_str_msg.data = obj_str
        self._publisher.publish(obj_str_msg)


@Publishers.register("Image", "ros2")
class ROS2ImagePublisher(ROS2Publisher):

    def __init__(self, name: str, out_port: str, carrier: str = "tcp", should_wait: bool = True, queue_size: int = QUEUE_SIZE,
                 width: int = -1, height: int = -1, rgb: bool = True, fp: bool = False, **kwargs):
        """
        The ImagePublisher using the ROS2 Image message assuming a numpy array as input

        :param name: str: Name of the publisher
        :param out_port: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS2 currently only supports TCP for pub/sub pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param width: int: Width of the image. Default is -1 meaning that the width is not fixed
        :param height: int: Height of the image. Default is -1 meaning that the height is not fixed
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        """
        super().__init__(name, out_port, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)

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
        """
        Establish the connection

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        self._publisher = self.create_publisher(sensor_msgs.msg.Image, self.out_port, qos_profile=self.queue_size)
        established = self.await_connection(self._publisher)
        return self.check_establishment(established)

    def publish(self, img):
        """
        Publish the image to the middleware

        :param img: np.ndarray: Image to publish formatted as a cv2 image np.ndarray[img_height, img_width, channels]
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)
        if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
            raise ValueError("Incorrect image shape for publisher")
        img = np.require(img, dtype=self._type, requirements='C')
        msg = sensor_msgs.msg.Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = self._encoding
        msg.is_bigendian = img.dtype.byteorder == '>' or (img.dtype.byteorder == '=' and sys.byteorder == 'big')
        msg.step = img.strides[0]
        msg.data = img.tobytes()
        self._publisher.publish(msg)


@Publishers.register("AudioChunk", "ros2")
class ROS2AudioChunkPublisher(ROS2Publisher):

    def __init__(self, name: str, out_port: str, carrier: str = "tcp", should_wait: bool = True, queue_size: int = QUEUE_SIZE,
                 channels: int = 1, rate: int = 44100, chunk: int = -1, **kwargs):
        """
        The AudioChunkPublisher using the ROS2 Image message assuming a numpy array as input

        :param name: str: Name of the publisher
        :param out_port: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS2 currently only supports TCP for pub/sub pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param channels: int: Number of channels. Default is 1
        :param rate: int: Sampling rate. Default is 44100
        :param chunk: int: Chunk size. Default is -1 meaning that the chunk size is not fixed
        """
        super().__init__(name, out_port, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)

        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._publisher = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        """
        Establish the connection

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        # self._publisher = rospy.Publisher(self.out_port, sensor_msgs.msg.Image, queue_size=self.queue_size)
        self._publisher = self.create_publisher(sensor_msgs.msg.Image, self.out_port, qos_profile=self.queue_size)
        established = self.await_connection(self._publisher)
        return self.check_establishment(established)

    def publish(self, aud: Tuple[np.ndarray, int]):
        """
        Publish the audio chunk to the middleware

        :param aud: (np.ndarray, int): Audio chunk to publish formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
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
        msg.header.stamp = self.get_clock().now().to_msg()
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
