import logging
import sys
import json
import time
import os
import base64
import io
import importlib.util
from typing import Optional, Tuple

import numpy as np
import cv2
import rospy
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapyfi.middlewares.ros import ROSMiddleware
from wrapyfi.encoders import JsonEncoder


QUEUE_SIZE = int(os.environ.get("WRAPYFI_ROS_QUEUE_SIZE", 5))
WATCHDOG_POLL_REPEAT = None


class ROSPublisher(Publisher):

    def __init__(self, name: str, out_topic: str, carrier: str = "tcp", should_wait: bool = True,
                 queue_size: int = QUEUE_SIZE, ros_kwargs: Optional[dict] = None, **kwargs):
        """
        Initialize the publisher

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param ros_kwargs: dict: Additional kwargs for the ROS middleware
        :param kwargs: dict: Additional kwargs for the publisher
        """
        if carrier or carrier != "tcp":
            logging.warning("[ROS] ROS does not support other carriers than TCP for PUB/SUB pattern. Using TCP.")
            carrier = "tcp"
        super().__init__(name, out_topic, carrier=carrier, should_wait=should_wait, **kwargs)
        ROSMiddleware.activate(**ros_kwargs or {})

        self.queue_size = queue_size

    def await_connection(self, publisher, out_topic: Optional[str] = None, repeats: Optional[int] = None):
        """
        Wait for at least one subscriber to connect to the publisher

        :param publisher: rospy.Publisher: Publisher to await connection to
        :param out_topic: str: Name of the output topic
        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        connected = False
        if out_topic is None:
            out_topic = self.out_topic
        logging.info(f"[ROS] Waiting for topic subscriber: {out_topic}")
        if repeats is None:
            if self.should_wait:
                repeats = -1
            else:
                repeats = 1
            while repeats > 0 or repeats <= -1:
                repeats -= 1
                connected = publisher.get_num_connections() < 1
                if connected:
                    break
                time.sleep(0.02)
        logging.info(f"[ROS] Topic subscriber connected: {out_topic}")
        return connected

    def close(self):
        """
        Close the publisher
        """
        if hasattr(self, "_publisher") and self._publisher:
            if self._publisher is not None:
                self._publisher.unregister()

    def __del__(self):
        self.close()


@Publishers.register("NativeObject", "ros")
class ROSNativeObjectPublisher(ROSPublisher):

    def __init__(self, name: str, out_topic: str, carrier: str = "tcp", should_wait: bool = True,
                 queue_size: int = QUEUE_SIZE, serializer_kwargs: Optional[dict] = None, **kwargs):
        """
        The NativeObject publisher using the ROS String message assuming a combination of python native objects

        and numpy arrays as input. Serializes the data (including plugins) using the encoder and sends it as a string
        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        """
        super().__init__(name, out_topic, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)
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
        self._publisher = rospy.Publisher(self.out_topic, std_msgs.msg.String, queue_size=self.queue_size)
        established = self.await_connection(self._publisher, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, obj):
        """
        Publish the object to the middlware

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
        self._publisher.publish(obj_str)


@Publishers.register("Image", "ros")
class ROSImagePublisher(ROSPublisher):

    def __init__(self, name: str, out_topic: str, carrier: str = "tcp",  should_wait: bool = True, queue_size: int = QUEUE_SIZE,
                 width: int = -1, height: int = -1, rgb: bool = True, fp: bool = False, jpg: bool = False, **kwargs):
        """
        The ImagePublisher using the ROS Image message assuming a numpy array as input

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param width: int: Width of the image. Default is -1 meaning that the width is not fixed
        :param height: int: Height of the image. Default is -1 meaning that the height is not fixed
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: bool: True if the image should be compressed as JPG. Default is False
        """
        super().__init__(name, out_topic, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)

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
        Establish the connection

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        if self.jpg:
            self._publisher = rospy.Publisher(self.out_topic, sensor_msgs.msg.CompressedImage, queue_size=self.queue_size)
        else:
            self._publisher = rospy.Publisher(self.out_topic, sensor_msgs.msg.Image, queue_size=self.queue_size)
        established = self.await_connection(self._publisher)
        return self.check_establishment(established)

    def publish(self, img):
        """
        Publish the image to the middleware

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
            img_msg.header.stamp = rospy.Time.now()
            img_msg.format = "jpeg"
            img_msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
        else:
            img_msg = sensor_msgs.msg.Image()
            img_msg.header.stamp = rospy.Time.now()
            img_msg.height = img.shape[0]
            img_msg.width = img.shape[1]
            img_msg.encoding = self._encoding
            img_msg.is_bigendian = img.dtype.byteorder == '>' or (img.dtype.byteorder == '=' and sys.byteorder == 'big')
            img_msg.step = img.strides[0]
            img_msg.data = img.tobytes()
        self._publisher.publish(img_msg)


@Publishers.register("AudioChunk", "ros")
class ROSAudioChunkPublisher(ROSPublisher):

    def __init__(self, name: str, out_topic: str, carrier: str = "tcp", should_wait: bool = True, queue_size: int = QUEUE_SIZE,
                 channels: int = 1, rate: int = 44100, chunk: int = -1, **kwargs):
        """
        The AudioChunkPublisher using the ROS Audio message assuming a numpy array as input

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        :param channels: int: Number of channels. Default is 1
        :param rate: int: Sampling rate. Default is 44100
        :param chunk: int: Chunk size. Default is -1 meaning that the chunk size is not fixed
        """
        super().__init__(name, out_topic, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)

        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._publisher = self._sound_msg = None
        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        """
        Establish the connection

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
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
        self._publisher = rospy.Publisher(self.out_topic, ROSAudioMessage, queue_size=self.queue_size)
        self._sound_msg = ROSAudioMessage()
        established = self.await_connection(self._publisher)
        return self.check_establishment(established)

    def publish(self, aud: Tuple[np.ndarray, int]):
        """
        Publish the audio chunk to the middleware

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
        if self.rate != -1 and rate != self.rate:
            raise ValueError("Incorrect audio rate for publisher")
        chunk, channels = aud.shape if len(aud.shape) > 1 else (aud.shape[0], 1)
        self.chunk = chunk if self.chunk == -1 else self.chunk
        self.channels = channels if self.channels == -1 else self.channels
        if (self.chunk != -1 and self.chunk != chunk) or (self.channels != -1 and self.channels != channels):
            raise ValueError("Incorrect audio shape for publisher")
        aud = np.require(aud, dtype=np.float32, requirements='C')

        aud_msg = self._sound_msg
        aud_msg.header.stamp = rospy.Time.now()
        aud_msg.chunk_size = chunk
        aud_msg.channels = channels
        aud_msg.sample_rate = rate
        aud_msg.is_bigendian = aud.dtype.byteorder == '>' or (aud.dtype.byteorder == '=' and sys.byteorder == 'big')
        aud_msg.encoding = 'S16BE' if aud_msg.is_bigendian else 'S16LE'
        aud_msg.step = aud.strides[0]
        aud_msg.data = aud.tobytes()  # (aud * 32767.0).tobytes()
        self._publisher.publish(aud_msg)


@Publishers.register("Properties", "ros")
class ROSPropertiesPublisher(ROSPublisher):
    """
    Sets rospy parameters. Behaves differently from other data types by directly setting ROS parameters.
    Note that the listener is not guaranteed to receive the updated signal, since the listener can trigger before
    property is set. The property decorated method returns accept native python objects (excluding None),
    but care should be taken when using dictionaries, since they are analogous with node namespaces:
    http://wiki.ros.org/rospy/Overview/Parameter%20Server
    """
    def __init__(self, name: str, out_topic: str, carrier: str = "tcp", persistent: bool = True, **kwargs):
        """
        The PropertiesPublisher using the ROS parameter server

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param persistent: bool: True if the parameter should be kept on closing node, False if it should be deleted or
                                    reset to its state before the node was started. Default is True
        """
        super().__init__(name, out_topic, carrier=carrier, **kwargs)
        self.persistent = persistent

        self.previous_property = False

    def establish(self, **kwargs):
        """
        Store the original property value in case it needs to be reset
        """
        self.previous_property = rospy.get_param(self.out_topic, False)

    def publish(self, obj):
        """
        Publish the property to the middleware (parameter server)

        :param obj: object: Property to publish. If dict, will be set as a namespace
        """
        rospy.set_param(self.out_topic, obj)

    def close(self):
        """
        Close the publisher and reset the property to its original value if not persistent
        """
        if hasattr(self, "out_topic") and not self.persistent:
            rospy.delete_param(self.out_topic)
            if self.previous_property:
                rospy.set_param(self.out_topic, self.previous_property)

    def __del__(self):
        self.close()


@Publishers.register("ROSMessage", "ros")
class ROSMessagePublisher(ROSPublisher):

    def __init__(self, name: str, out_topic: str, carrier: str = "tcp",
                 should_wait: bool = True, queue_size: int = QUEUE_SIZE, **kwargs):
        """
        The ROSMessagePublisher using the ROS message type inferred from the message type. Supports standard ROS msgs

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param queue_size: int: Queue size for the publisher. Default is 5
        """
        super().__init__(name, out_topic, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)

        self._publisher = None
        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, obj=None, **kwargs):
        """
        Establish the connection

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :param obj: object: Object to establish the connection to
        :return: bool: True if connection established, False otherwise
        """
        if obj is None:
            return
        obj_type = obj._type.split("/")
        import_msg = importlib.import_module(f"{obj_type[0]}.msg")
        msg_type = getattr(import_msg, obj_type[1])
        self._publisher = rospy.Publisher(self.out_topic, msg_type, queue_size=self.queue_size)
        established = self.await_connection(self._publisher, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, obj):
        """
        Publish the object to the middleware

        :param obj: object: ROS message to publish
        """
        if not self.established:
            established = self.establish(obj=obj)
            if not established:
                return
            else:
                time.sleep(0.2)
        self._publisher.publish(obj)
