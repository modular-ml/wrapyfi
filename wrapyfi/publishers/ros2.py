import logging
import sys
import json
import time
import os
import importlib
from typing import Optional, Tuple

import numpy as np
import cv2
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

    def __init__(self, name: str, out_topic: str, should_wait: bool = True,
                 queue_size: int = QUEUE_SIZE, ros2_kwargs: Optional[dict] = None, **kwargs):
        """
        Initialize the publisher.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param ros2_kwargs: dict: Additional kwargs for the ROS2 middleware
        :param kwargs: dict: Additional kwargs for the publisher
        """
        carrier = "tcp"
        if "carrier" in kwargs and kwargs["carrier"] not in ["", None]:
            logging.warning("[ROS2] ROS2 currently does not support explicit carrier setting for PUB/SUB pattern. Using TCP.")
        if "carrier" in kwargs:
            del kwargs["carrier"]
        ROS2Middleware.activate(**ros2_kwargs or {})
        Publisher.__init__(self, name, out_topic, carrier=carrier, should_wait=should_wait, **kwargs)
        Node.__init__(self, name + str(hex(id(self))))

        self.queue_size = queue_size

    def await_connection(self, publisher, out_topic: Optional[str] = None, repeats: Optional[int] = None):
        """
        Wait for at least one subscriber to connect to the publisher.

        :param publisher: rclpy.publisher.Publisher: Publisher to await connection to
        :param out_topic: str: Name of the output topic
        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        connected = False
        if out_topic is None:
            out_topic = self.out_topic
        logging.info(f"[ROS2] Waiting for topic subscriber: {out_topic}")
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
        logging.info(f"[ROS2] Topic subscriber connected: {out_topic}")
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

    def __init__(self, name, out_topic: str, should_wait: bool = True,
                 queue_size: int = QUEUE_SIZE, serializer_kwargs: Optional[dict] = None, **kwargs):
        """
        The NativeObject publisher using the ROS2 String message assuming a combination of python native objects
        and numpy arrays as input. Serializes the data (including plugins) using the encoder and sends it as a string.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        """
        super().__init__(name, out_topic, should_wait=should_wait, queue_size=queue_size, **kwargs)
        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}

        self._publisher = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        self._publisher = self.create_publisher(std_msgs.msg.String, self.out_topic, qos_profile=self.queue_size)
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

    def __init__(self, name: str, out_topic: str, should_wait: bool = True, queue_size: int = QUEUE_SIZE,
                 width: int = -1, height: int = -1, rgb: bool = True, fp: bool = False, jpg: bool = False, **kwargs):
        """
        The ImagePublisher using the ROS2 Image message assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param width: int: Width of the image. Default is -1 meaning that the width is not fixed
        :param height: int: Height of the image. Default is -1 meaning that the height is not fixed
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: bool: True if the image should be compressed as JPG. Default is False
        """
        super().__init__(name, out_topic, should_wait=should_wait, queue_size=queue_size, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        if self.fp:
            self._encoding = '32FC3' if self.rgb else '32FC1'
            self._type = np.float32
        else:
            self._encoding = 'bgr8' if self.rgb else 'mono8'
            self._type = np.uint8
        if self.jpg:
            self._encoding = 'jpeg'
            self._type = np.uint8

        self._publisher = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        if self.jpg:
            self._publisher = self.create_publisher(sensor_msgs.msg.CompressedImage, self.out_topic, qos_profile=self.queue_size)
        else:
            self._publisher = self.create_publisher(sensor_msgs.msg.Image, self.out_topic, qos_profile=self.queue_size)
        established = self.await_connection(self._publisher)
        return self.check_establishment(established)

    def publish(self, img: np.ndarray):
        """
        Publish the image to the middleware.

        :param img: np.ndarray: Image to publish formatted as a cv2 image np.ndarray[img_height, img_width, channels]
        """
        if img is None:
            return

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

        if self.jpg:
            img_msg = sensor_msgs.msg.CompressedImage()
            img_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
            img_msg.format = "jpeg"
            img_msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
        else:
            img_msg = sensor_msgs.msg.Image()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.height = img.shape[0]
            img_msg.width = img.shape[1]
            img_msg.encoding = self._encoding
            img_msg.is_bigendian = img.dtype.byteorder == '>' or (img.dtype.byteorder == '=' and sys.byteorder == 'big')
            img_msg.step = img.strides[0]
            img_msg.data = img.tobytes()
        self._publisher.publish(img_msg)


@Publishers.register("AudioChunk", "ros2")
class ROS2AudioChunkPublisher(ROS2Publisher):

    def __init__(self, name: str, out_topic: str, should_wait: bool = True, queue_size: int = QUEUE_SIZE,
                 channels: int = 1, rate: int = 44100, chunk: int = -1, **kwargs):
        """
        The AudioChunkPublisher using the ROS2 Audio message assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param channels: int: Number of channels. Default is 1
        :param rate: int: Sampling rate. Default is 44100
        :param chunk: int: Chunk size. Default is -1 meaning that the chunk size is not fixed
        """
        super().__init__(name, out_topic, should_wait=should_wait, queue_size=queue_size, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._publisher = self._sound_msg = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        try:
            from wrapyfi_ros2_interfaces.msg import ROS2AudioMessage
        except ImportError:
            import wrapyfi
            logging.error("[ROS2] Could not import ROS2AudioMessage. "
                          "Make sure the ROS2 services in wrapyfi_extensions/wrapyfi_ros2_interfaces are compiled. "
                          "Refer to the documentation for more information: \n" +
                          wrapyfi.__url__ + "wrapyfi_extensions/wrapyfi_ros2_interfaces/README.md")
            sys.exit(1)
        self._publisher = self.create_publisher(ROS2AudioMessage, self.out_topic, qos_profile=self.queue_size)
        self._sound_msg = ROS2AudioMessage()
        established = self.await_connection(self._publisher)
        return self.check_establishment(established)

    def publish(self, aud: Tuple[np.ndarray, int]):
        """
        Publish the audio chunk to the middleware.

        :param aud: Tuple[np.ndarray, int]: Audio chunk to publish formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)

        aud, rate = aud
        if aud is None:
            return
        if 0 < self.rate != rate:
            raise ValueError("Incorrect audio rate for publisher")
        chunk, channels = aud.shape if len(aud.shape) > 1 else (aud.shape[0], 1)
        self.chunk = chunk if self.chunk == -1 else self.chunk
        self.channels = channels if self.channels == -1 else self.channels
        if 0 < self.chunk != chunk or 0 < self.channels != channels:
            raise ValueError("Incorrect audio shape for publisher")
        aud = np.require(aud, dtype=np.float32, requirements='C')

        aud_msg = self._sound_msg
        aud_msg.header.stamp = self.get_clock().now().to_msg()
        aud_msg.chunk_size = chunk
        aud_msg.channels = channels
        aud_msg.sample_rate = rate
        aud_msg.is_bigendian = aud.dtype.byteorder == '>' or (aud.dtype.byteorder == '=' and sys.byteorder == 'big')
        aud_msg.encoding = 'S16BE' if aud_msg.is_bigendian else 'S16LE'
        aud_msg.step = aud.strides[0]
        aud_msg.data = aud.tobytes()  # (aud * 32767.0).tobytes()
        self._publisher.publish(aud_msg)


@Publishers.register("Properties", "ros2")
class ROS2PropertiesPublisher(ROS2Publisher):
    def __init__(self, name, out_topic, **kwargs):
        super().__init__(name, out_topic, **kwargs)
        raise NotImplementedError


@Publishers.register("ROS2Message", "ros2")
class ROS2MessagePublisher(ROS2Publisher):

    def __init__(self, name: str, out_topic: str, should_wait: bool = True, queue_size: int = QUEUE_SIZE, **kwargs):
        """
        The ROS2MessagePublisher using the ROS2 message type determined dynamically.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        """
        super().__init__(name, out_topic, should_wait=should_wait, queue_size=queue_size, **kwargs)
        self._publisher = None
        self._msg_type = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def get_message_type(self, msg):
        """
        Get the type of a specific message.

        :param msg: ROS2 message object
        :return: type: The type of the provided message
        """
        return type(msg)

    def establish(self, msg, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection using the provided message to determine the type.

        :param msg: ROS2Message: Message to determine the type.
        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        self._msg_type = self.get_message_type(msg)

        self._publisher = self.create_publisher(self._msg_type, self.out_topic, qos_profile=self.queue_size)
        return self.await_connection(self._publisher)

    def publish(self, msg):
        """
        Publish the data to the middleware.

        :param msg: ROS2Message: Message to publish. This should be formatted according to the expected message type.
        """
        if not self._publisher:
            self.establish(msg)

        if not self.established:
            established = self.establish(msg, repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)

        self._publisher.publish(msg)
