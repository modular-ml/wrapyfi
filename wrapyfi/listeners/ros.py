import logging
import sys
import json
import queue
import time
import os
from typing import Optional, Any

import numpy as np
import cv2
import rospy
import rostopic
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.listeners import Listener, Listeners, ListenerWatchDog
from wrapyfi.middlewares.ros import ROSMiddleware
from wrapyfi.encoders import JsonDecodeHook


QUEUE_SIZE = int(os.environ.get("WRAPYFI_ROS_QUEUE_SIZE", 5))
WATCHDOG_POLL_REPEAT = None


class ROSListener(Listener):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", should_wait: bool = True,
                 queue_size: int = QUEUE_SIZE, ros_kwargs: Optional[dict] = None, **kwargs):
        """
        Initialize the subscriber

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        :param ros_kwargs: dict: Additional kwargs for the ROS middleware
        :param kwargs: dict: Additional kwargs for the subscriber
        """
        if carrier or carrier != "tcp":
            logging.warning("[ROS] ROS does not support other carriers than TCP for PUB/SUB pattern. Using TCP.")
            carrier = "tcp"
        super().__init__(name, in_topic, carrier=carrier, should_wait=should_wait, **kwargs)
        ROSMiddleware.activate(**ros_kwargs or {})
        
        self.queue_size = queue_size

    def close(self):
        """
        Close the subscriber
        """
        if hasattr(self, "_subscriber") and self._subscriber:
            if self._subscriber is not None:
                self._subscriber.unregister()

    def __del__(self):
        self.close()


@Listeners.register("NativeObject", "ros")
class ROSNativeObjectListener(ROSListener):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", should_wait: bool = True, queue_size: int =QUEUE_SIZE,
                 deserializer_kwargs: Optional[dict] = None, **kwargs):
        """
        The NativeObject listener using the ROS String message assuming the data is serialized as a JSON string.
        Deserializes the data (including plugins) using the decoder and parses it to a Python object

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, in_topic, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)

        self._subscriber = self._queue = None

        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        ListenerWatchDog().add_listener(self)

    def establish(self):
        """
        Establish the subscriber
        """
        self._queue = queue.Queue(maxsize=0 if self.queue_size is None or self.queue_size <= 0 else self.queue_size)
        self._subscriber = rospy.Subscriber(self.in_topic, std_msgs.msg.String, callback=self._message_callback)
        self.established = True

    def listen(self) -> Any:
        """
        Listen for a message

        :return: Any: The received message as a native python object
        """
        if not self.established:
            self.establish()
        try:
            obj_str = self._queue.get(block=self.should_wait)
            return json.loads(obj_str, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        except queue.Empty:
            return None

    def _message_callback(self, msg):
        """
        Callback for the subscriber

        :param msg: std_msgs.msg.String: The received message
        """
        try:
            self._queue.put(msg.data, block=False)
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because listener queue is full: {self.in_topic}")


@Listeners.register("Image", "ros")
class ROSImageListener(ROSListener):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", should_wait: bool = True, queue_size: int = QUEUE_SIZE,
                 width: int = -1, height: int = -1, rgb: bool = True, fp: bool = False, jpg: bool = False, **kwargs):
        """
        The Image listener using the ROS Image message parsed to a numpy array

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        :param width: int: Width of the image. Default is -1 (use the width of the received image)
        :param height: int: Height of the image. Default is -1 (use the height of the received image)
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: bool: True if the image should be decompressed from JPG. Default is False
        """
        super().__init__(name, in_topic, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)

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

        self._pixel_bytes = (3 if self.rgb else 1) * np.dtype(self._type).itemsize

        self._subscriber = self._queue = None

        ListenerWatchDog().add_listener(self)

    def establish(self):
        """
        Establish the subscriber
        """
        self._queue = queue.Queue(maxsize=0 if self.queue_size is None or self.queue_size <= 0 else self.queue_size)
        if self.jpg:
            self._subscriber = rospy.Subscriber(self.in_topic, sensor_msgs.msg.CompressedImage, callback=self._message_callback)
        else:
            self._subscriber = rospy.Subscriber(self.in_topic, sensor_msgs.msg.Image, callback=self._message_callback)
        self.established = True

    def listen(self):
        """
        Listen for a message

        :return: np.ndarray: The received message as a numpy array formatted as a cv2 image np.ndarray[img_height, img_width, channels]
        """
        if not self.established:
            self.establish()
        try:
            if self.jpg:
                format, data = self._queue.get(block=self.should_wait)
                if format != 'jpeg':
                    raise ValueError(f"Unsupported image format: {format}")
                if self.rgb:
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                else:
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            else:
                height, width, encoding, is_bigendian, data = self._queue.get(block=self.should_wait)
                if encoding != self._encoding:
                    raise ValueError("Incorrect encoding for listener")
                elif 0 < self.width != width or 0 < self.height != height or len(data) != height * width * self._pixel_bytes:
                    raise ValueError("Incorrect image shape for listener")
                img = np.frombuffer(data, dtype=np.dtype(self._type).newbyteorder('>' if is_bigendian else '<')).reshape((height, width, -1))
                if img.shape[2] == 1:
                    img = img.squeeze(axis=2)
            return img
        except queue.Empty:
            return None

    def _message_callback(self, data):
        """
        Callback for the subscriber

        :param data: sensor_msgs.msg.Image: The received message
        """
        try:
            if self.jpg:
                self._queue.put((data.format, data.data), block=False)
            else:
                self._queue.put((data.height, data.width, data.encoding, data.is_bigendian, data.data), block=False)
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because listener queue is full: {self.in_topic}")


@Listeners.register("AudioChunk", "ros")
class ROSAudioChunkListener(ROSListener):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", should_wait: bool = True, queue_size: int = QUEUE_SIZE,
                 channels: int = 1, rate: int = 44100, chunk: int = -1, **kwargs):
        """
        The AudioChunk listener using the ROS Image message parsed to a numpy array

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        :param channels: int: Number of channels in the audio. Default is 1
        :param rate: int: Sampling rate of the audio. Default is 44100
        :param chunk: int: Number of samples in the audio chunk. Default is -1 (use the chunk size of the received audio)
        """
        super().__init__(name, in_topic, carrier=carrier, should_wait=should_wait, queue_size=queue_size,
                         width=chunk, height=channels, rgb=False, fp=True, jpg=False, **kwargs)

        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._subscriber = self._queue = None
        ListenerWatchDog().add_listener(self)

    def establish(self):
        """
        Establish the subscriber
        """
        try:
            from wrapyfi_ros_interfaces.msg import ROSAudioMessage
        except ImportError:
            import wrapyfi
            logging.error("[ROS] Could not import ROSAudioMessage. "
                          "Make sure the ROS messages in wrapyfi_extensions/wrapyfi_ros_interfaces are compiled. "
                          "Refer to the documentation for more information: \n" +
                          wrapyfi.__url__ + "wrapyfi_extensions/wrapyfi_ros_interfaces/README.md")
            sys.exit(1)
        self._queue = queue.Queue(maxsize=0 if self.queue_size is None or self.queue_size <= 0 else self.queue_size)
        self._subscriber = rospy.Subscriber(self.in_topic, ROSAudioMessage, callback=self._message_callback)
        self.established = True

    def listen(self):
        """
        Listen for a message

        :return: Tuple[np.ndarray, int]: The received message as a numpy array formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        if not self.established:
            self.establish()
        try:
            chunk, channels, rate, encoding, is_bigendian, data = self._queue.get(block=self.should_wait)
            if self.rate != -1 and rate != self.rate:
                raise ValueError("Incorrect audio rate for listener")
            if encoding not in ['S16LE', 'S16BE']:
                raise ValueError("Incorrect encoding for listener")
            elif 0 < self.chunk != chunk or self.channels != channels or len(data) != chunk * channels * 4:
                raise ValueError("Incorrect audio shape for listener")
            aud = np.frombuffer(data, dtype=np.dtype(np.float32).newbyteorder('>' if is_bigendian else '<')).reshape((chunk, channels))
            # aud = aud / 32767.0
            if aud.shape[1] == 1:
                aud = np.squeeze(aud)
            return aud, rate
        except queue.Empty:
            return None, self.rate

    def _message_callback(self, data):
        """
        Callback for the subscriber
        :param data: wrapyfi_ros_interfaces.msg.ROSAudioMessage: The received message
        """
        try:
            self._queue.put((data.chunk_size, data.channels, data.sample_rate, data.encoding, data.is_bigendian, data.data), block=False)
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because listener queue is full: {self.in_topic}")


@Listeners.register("Properties", "ros")
class ROSPropertiesListener(ROSListener):
    """
    Gets rospy parameters. Behaves differently from other data types by directly acquiring ROS parameters.
    Note that the listener is not guaranteed to receive the updated signal, since the listener can trigger before
    property is set. The property decorated method returns accept native python objects (excluding None),
    but care should be taken when using dictionaries, since they are analogous with node namespaces:
    http://wiki.ros.org/rospy/Overview/Parameter%20Server
    """
    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", should_wait: bool = True, queue_size: int = QUEUE_SIZE, **kwargs):
        """
        The PropertiesListener using the ROS Parameter Server

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether the subscriber should wait for a parameter to be set. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        """
        super().__init__(name, in_topic, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)
        self._subscriber = self._queue = None

        if not self.should_wait:
            ListenerWatchDog().add_listener(self)

        self.previous_property = False

    def await_connection(self, in_topic: Optional[int] = None, repeats: Optional[int] = None):
        """
        Wait for a parameter to be set

        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param repeats: int: Number of times to check for the parameter. None for infinite. Default is None
        """
        connected = False
        if in_topic is None:
            in_topic = self.in_topic
        logging.info(f"[ROS] Waiting for property: {in_topic}")
        if repeats is None:
            if self.should_wait:
                repeats = -1
            else:
                repeats = 1

            while repeats > 0 or repeats <= -1:
                repeats -= 1
                self.previous_property = rospy.get_param(self.in_topic, False)
                connected = True if self.previous_property else False
                if connected:
                    logging.info(f"[ROS] Found property: {in_topic}")
                    break
                time.sleep(0.2)
        return connected

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the subscriber

        :param repeats: int: Number of times to check for the parameter. None for infinite. Default is None
        """
        established = self.await_connection(repeats=repeats)
        return self.check_establishment(established)

    def listen(self):
        """
        Listen for a message

        :return: Any: The received message as a native python object
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                obj = None
            else:
                obj = self.previous_property
            return obj
        else:
            obj = rospy.get_param(self.in_topic, False)
            return obj


@Listeners.register("ROSMessage", "ros")
class ROSMessageListener(ROSListener):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", should_wait: bool = True, queue_size: int = QUEUE_SIZE, **kwargs):
        """
        The ROSMessageListener using the ROS message type inferred from the message type. Supports standard ROS msgs

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether the subscriber should wait for a message to be published. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        """
        super().__init__(name, in_topic, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)

        self._subscriber = self._queue = self._topic_type = None

        ListenerWatchDog().add_listener(self)

    def establish(self):
        """
        Establish the subscriber
        """
        self._queue = queue.Queue(maxsize=0 if self.queue_size is None or self.queue_size <= 0 else self.queue_size)
        self._topic_type, topic_str, _ = rostopic.get_topic_class(self.in_topic, blocking=self.should_wait)
        if self._topic_type is None:
            return
        self._subscriber = rospy.Subscriber(self.in_topic, self._topic_type, callback=self._message_callback)
        self.established = True

    def listen(self):
        """
        Listen for a message

        :return: rospy.msg: The received message as a ROS message object
        """
        if not self.established:
            self.establish()
        try:
            obj = self._queue.get(block=self.should_wait)

            return obj  # self._topic_type.deserialize_numpy(obj_str)
        except queue.Empty:
            return None

    def _message_callback(self, msg):
        """
        Callback for the subscriber

        :param msg: rospy.msg: The received message as a ROS message object
        """
        try:
            self._queue.put(msg, block=False)
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because listener queue is full: {self.in_topic}")

