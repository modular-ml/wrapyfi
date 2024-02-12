import logging
import json
import queue
import time
import os
from typing import Optional
import sys
import importlib

import numpy as np
import cv2
import rclpy
from rclpy import Parameter
from rclpy.node import Node
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.listeners import Listener, Listeners, ListenerWatchDog
from wrapyfi.middlewares.ros2 import ROS2Middleware
from wrapyfi.encoders import JsonDecodeHook


WAIT = {True: None, False: 0}
WATCHDOG_POLL_REPEAT = None
QUEUE_SIZE = int(os.environ.get("WRAPYFI_ROS2_QUEUE_SIZE", 5))


class ROS2Listener(Listener, Node):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        queue_size: int = QUEUE_SIZE,
        ros2_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the subscriber.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        :param ros2_kwargs: dict: Additional kwargs for the ROS 2 middleware
        :param kwargs: dict: Additional kwargs for the subscriber
        """
        carrier = "tcp"
        if "carrier" in kwargs and kwargs["carrier"] not in ["", None]:
            logging.warning(
                "[ROS 2] ROS 2 currently does not support explicit carrier setting for PUB/SUB pattern. Using TCP."
            )
        if "carrier" in kwargs:
            del kwargs["carrier"]

        ROS2Middleware.activate(**ros2_kwargs or {})
        Listener.__init__(
            self, name, in_topic, carrier=carrier, should_wait=should_wait, **kwargs
        )
        Node.__init__(self, name + str(hex(id(self))), allow_undeclared_parameters=True)

        self.queue_size = queue_size

    def close(self):
        """
        Close the subscriber.
        """
        if hasattr(self, "_subscriber") and self._subscriber:
            if self._subscriber is not None:
                self.destroy_node()

    def __del__(self):
        self.close()


@Listeners.register("NativeObject", "ros2")
class ROS2NativeObjectListener(ROS2Listener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        queue_size: int = QUEUE_SIZE,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The NativeObject listener using the ROS 2 String message assuming the data is serialized as a JSON string.
        Deserializes the data (including plugins) using the decoder and parses it to a native object.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(
            name, in_topic, should_wait=should_wait, queue_size=queue_size, **kwargs
        )

        self._subscriber = self._queue = None

        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        ListenerWatchDog().add_listener(self)

    def establish(self):
        """
        Establish the subscriber.
        """
        self._queue = queue.Queue(
            maxsize=(
                0
                if self.queue_size is None or self.queue_size <= 0
                else self.queue_size
            )
        )
        self._subscriber = self.create_subscription(
            std_msgs.msg.String,
            self.in_topic,
            callback=self._message_callback,
            qos_profile=self.queue_size,
        )
        self.established = True

    def listen(self):
        """
        Listen for a message.

        :return: Any: The received message as a native python object
        """
        if not self.established:
            self.establish()
        try:
            rclpy.spin_once(self, timeout_sec=WAIT[self.should_wait])
            obj_str = self._queue.get(block=self.should_wait)
            return json.loads(
                obj_str,
                object_hook=self._plugin_decoder_hook,
                **self._deserializer_kwargs,
            )
        except queue.Empty:
            return None

    def _message_callback(self, msg):
        """
        Callback for the subscriber.

        :param msg: std_msgs.msg.String: The received message
        """
        try:
            self._queue.put(msg.data, block=False)
        except queue.Full:
            logging.warning(
                f"[ROS 2] Discarding data because listener queue is full: {self.in_topic}"
            )


@Listeners.register("Image", "ros2")
class ROS2ImageListener(ROS2Listener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        queue_size: int = QUEUE_SIZE,
        width: int = -1,
        height: int = -1,
        rgb: bool = True,
        fp: bool = False,
        jpg: bool = False,
        **kwargs,
    ):
        """
        The Image listener using the ROS 2 Image message parsed to a numpy array.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        :param width: int: Width of the image. Default is -1 (use the width of the received image)
        :param height: int: Height of the image. Default is -1 (use the height of the received image)
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: bool: True if the image should be decompressed from JPG. Default is False
        """
        super().__init__(
            name, in_topic, should_wait=should_wait, queue_size=queue_size, **kwargs
        )
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        if self.fp:
            self._encoding = "32FC3" if self.rgb else "32FC1"
            self._type = np.float32
        else:
            self._encoding = "bgr8" if self.rgb else "mono8"
            self._type = np.uint8
        if self.jpg:
            self._encoding = "jpeg"
            self._type = np.uint8

        self._pixel_bytes = (3 if self.rgb else 1) * np.dtype(self._type).itemsize

        self._subscriber = self._queue = None

        ListenerWatchDog().add_listener(self)

    def establish(self):
        """
        Establish the subscriber
        """
        self._queue = queue.Queue(
            maxsize=(
                0
                if self.queue_size is None or self.queue_size <= 0
                else self.queue_size
            )
        )
        if self.jpg:
            self._subscriber = self.create_subscription(
                sensor_msgs.msg.CompressedImage,
                self.in_topic,
                callback=self._message_callback,
                qos_profile=self.queue_size,
            )
        else:
            self._subscriber = self.create_subscription(
                sensor_msgs.msg.Image,
                self.in_topic,
                callback=self._message_callback,
                qos_profile=self.queue_size,
            )
        self.established = True

    def listen(self):
        """
        Listen for a message.

        :return: np.ndarray: The received message as a numpy array formatted as a cv2 image np.ndarray[img_height, img_width, channels]
        """
        if not self.established:
            self.establish()
        try:
            rclpy.spin_once(self, timeout_sec=WAIT[self.should_wait])
            if self.jpg:
                format, data = self._queue.get(block=self.should_wait)
                if format != "jpeg":
                    raise ValueError(f"Unsupported image format: {format}")
                if self.rgb:
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                else:
                    img = cv2.imdecode(
                        np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE
                    )
            else:
                height, width, encoding, is_bigendian, data = self._queue.get(
                    block=self.should_wait
                )
                if encoding != self._encoding:
                    raise ValueError("Incorrect encoding for listener")
                if (
                    0 < self.width != width
                    or 0 < self.height != height
                    or len(data) != height * width * self._pixel_bytes
                ):
                    raise ValueError("Incorrect image shape for listener")
                img = np.frombuffer(
                    data,
                    dtype=np.dtype(self._type).newbyteorder(
                        ">" if is_bigendian else "<"
                    ),
                ).reshape((height, width, -1))
                if img.shape[2] == 1:
                    img = img.squeeze(axis=2)
            return img
        except queue.Empty:
            return None

    def _message_callback(self, msg):
        """
        Callback for the subscriber.

        :param msg: sensor_msgs.msg.Image: The received message
        """
        try:
            if self.jpg:
                self._queue.put((msg.format, msg.data), block=False)
            else:
                self._queue.put(
                    (msg.height, msg.width, msg.encoding, msg.is_bigendian, msg.data),
                    block=False,
                )
        except queue.Full:
            logging.warning(
                f"[ROS 2] Discarding data because listener queue is full: {self.in_topic}"
            )


@Listeners.register("AudioChunk", "ros2")
class ROS2AudioChunkListener(ROS2Listener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        queue_size: int = QUEUE_SIZE,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = -1,
        **kwargs,
    ):
        """
        The AudioChunk listener using the ROS 2 Audio message parsed to a numpy array.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        :param channels: int: Number of channels in the audio. Default is 1
        :param rate: int: Sampling rate of the audio. Default is 44100
        :param chunk: int: Number of samples in the audio chunk. Default is -1 (use the chunk size of the received audio)
        """
        super().__init__(
            name, in_topic, should_wait=should_wait, queue_size=queue_size, **kwargs
        )
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._subscriber = self._queue = None

        ListenerWatchDog().add_listener(self)

    def establish(self):
        """
        Establish the subscriber.
        """
        try:
            from wrapyfi_ros2_interfaces.msg import ROS2AudioMessage
        except ImportError:
            import wrapyfi

            logging.error(
                "[ROS 2] Could not import ROS2AudioMessage. "
                "Make sure the ROS 2 services in wrapyfi_extensions/wrapyfi_ros2_interfaces are compiled. "
                "Refer to the documentation for more information: \n"
                + wrapyfi.__doc__
                + "ros2_interfaces_lnk.html"
            )
            sys.exit(1)
        self._queue = queue.Queue(
            maxsize=(
                0
                if self.queue_size is None or self.queue_size <= 0
                else self.queue_size
            )
        )
        self._subscriber = self.create_subscription(
            ROS2AudioMessage,
            self.in_topic,
            callback=self._message_callback,
            qos_profile=self.queue_size,
        )
        self.established = True

    def listen(self):
        """
        Listen for a message.

        :return: Tuple[np.ndarray, int]: The received message as a numpy array formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        if not self.established:
            self.establish()
        try:
            rclpy.spin_once(self, timeout_sec=WAIT[self.should_wait])
            chunk, channels, rate, encoding, is_bigendian, data = self._queue.get(
                block=self.should_wait
            )
            if 0 < self.rate != rate:
                raise ValueError("Incorrect audio rate for publisher")
            if encoding not in ["S16LE", "S16BE"]:
                raise ValueError("Incorrect encoding for listener")
            if (
                0 < self.chunk != chunk
                or self.channels != channels
                or len(data) != chunk * channels * 4
            ):
                raise ValueError("Incorrect audio shape for listener")
            aud = np.frombuffer(
                data,
                dtype=np.dtype(np.float32).newbyteorder(">" if is_bigendian else "<"),
            ).reshape((chunk, channels))
            # aud = aud / 32767.0
            return aud, rate
        except queue.Empty:
            return None, self.rate

    def _message_callback(self, msg):
        """
        Callback for the subscriber.

        :param msg: wrapyfi_ros2_interfaces.msg.ROS2AudioMessage: The received message
        """
        try:
            self._queue.put(
                (
                    msg.chunk_size,
                    msg.channels,
                    msg.sample_rate,
                    msg.encoding,
                    msg.is_bigendian,
                    msg.data,
                ),
                block=False,
            )
        except queue.Full:
            logging.warning(
                f"[ROS 2] Discarding data because listener queue is full: {self.in_topic}"
            )


@Listeners.register("Properties", "ros2")
class ROS2PropertiesListener(ROS2Listener):
    def __init__(self, name, in_topic, **kwargs):
        super().__init__(name, in_topic, **kwargs)
        raise NotImplementedError


@Listeners.register("ROS2Message", "ros2")
class ROS2MessageListener(ROS2Listener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        queue_size: int = QUEUE_SIZE,
        **kwargs,
    ):
        """
        The ROS2MessageListener using the ROS 2 message type inferred from the message type. Supports standard ROS 2 msgs.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        """
        super().__init__(
            name, in_topic, should_wait=should_wait, queue_size=queue_size, **kwargs
        )
        self._queue = queue.Queue(
            maxsize=(
                0
                if self.queue_size is None or self.queue_size <= 0
                else self.queue_size
            )
        )

    def get_topic_type(self, topic_name):
        """
        Get the type of a specific topic.

        :param topic_name: str: Name of the topic to get its type
        :return: str or None: The topic type as a string, or None if the topic does not exist
        """
        topic_names_and_types = self.get_topic_names_and_types()
        for name, types in topic_names_and_types:
            if name == topic_name:
                return types[0]
        return None

    def establish(self):
        """
        Establish the subscriber.
        """
        while True:
            topic_type_str = self.get_topic_type(self.in_topic)
            if not self.should_wait:
                break
            if topic_type_str:
                break
        if not topic_type_str:
            return None

        module_name, class_name = topic_type_str.rsplit("/", 1)
        module_name = module_name.replace("/", ".")
        MessageType = getattr(importlib.import_module(module_name), class_name)

        self._subscriber = self.create_subscription(
            MessageType,
            self.in_topic,
            callback=self._message_callback,
            qos_profile=self.queue_size,
        )
        self.established = True

    def listen(self):
        """
        Listen for a message.

        :return: ROS2Message: The received message as a ROS 2 message object
        """
        if not self.established:
            self.establish()
        try:
            rclpy.spin_once(self, timeout_sec=WAIT[self.should_wait])
            data = self._queue.get(block=self.should_wait)

            return data
        except queue.Empty:
            return None

    def _message_callback(self, msg):
        """
        Callback for the subscriber.

        :param msg: ROS2Message: The received message
        """
        try:
            self._queue.put(msg, block=False)
        except queue.Full:
            logging.warning(
                f"[ROS 2] Discarding data because listener queue is full: {self.in_topic}"
            )
