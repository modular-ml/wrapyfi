import logging
import sys
import json
import time
import os
import importlib.util
import queue
from typing import Optional, Tuple

import numpy as np
import rospy
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.servers import Server, Servers
from wrapyfi.middlewares.ros import ROSMiddleware, ROSNativeObjectService, ROSImageService
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class ROSServer(Server):

    def __init__(self, name: str, out_topic: str, carrier: str = "tcp", ros_kwargs: Optional[dict] = None, **kwargs):
        """
        Initialize the server.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for REQ/REP pattern. Default is 'tcp'
        :param ros_kwargs: dict: Additional kwargs for the ROS middleware
        :param kwargs: dict: Additional kwargs for the server
        """
        if carrier or carrier != "tcp":
            logging.warning("[ROS] ROS does not support other carriers than TCP for REQ/REP pattern. Using TCP.")
            carrier = "tcp"
        super().__init__(name, out_topic, carrier=carrier, **kwargs)
        ROSMiddleware.activate(**ros_kwargs or {})

    def close(self):
        """
        Close the server.
        """
        if hasattr(self, "_server") and self._server:
            if self._server is not None:
                self._server.shutdown()

    def __del__(self):
        self.close()


@Servers.register("NativeObject", "ros")
class ROSNativeObjectServer(ROSServer):
    SEND_QUEUE = queue.Queue(maxsize=1)
    RECEIVE_QUEUE = queue.Queue(maxsize=1)

    def __init__(self, name: str, out_topic: str, carrier: str = "tcp",
                 serializer_kwargs: Optional[dict] = None, deserializer_kwargs: Optional[dict] = None, **kwargs):
        """
        Specific server handling native Python objects, serializing them to JSON strings for transmission.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for REQ/REP pattern. Default is 'tcp'
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, out_topic, carrier=carrier, **kwargs)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._server = None

    def establish(self):
        """
        Establish the connection to the server.
        """
        self._server = rospy.Service(self.out_topic, ROSNativeObjectService, self._service_callback)
        self.established = True

    def await_request(self, *args, **kwargs):
        """
        Await and deserialize the client's request, returning the extracted arguments and keyword arguments.
        The method blocks until a message is received, then attempts to deserialize it using the configured JSON decoder
        hook, returning the extracted arguments and keyword arguments.

        :return: Tuple[list, dict]: A tuple containing two items:
                    - A list of arguments extracted from the received message
                    - A dictionary of keyword arguments extracted from the received message
        """
        if not self.established:
            self.establish()
        try:
            request = ROSNativeObjectServer.RECEIVE_QUEUE.get(block=True)
            [args, kwargs] = json.loads(request.data, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
            return args, kwargs
        except rospy.ServiceException as e:
            logging.error("[ROS] Service call failed: %s" % e)
            return [], {}

    @staticmethod
    def _service_callback(msg):
        """
        Callback for the ROS service.

        :param msg: ROSNativeObjectService._request_class: The request message
        :return: ROSNativeObjectService._response_class: The response message
        """
        ROSNativeObjectServer.RECEIVE_QUEUE.put(msg)
        return ROSNativeObjectServer.SEND_QUEUE.get(block=True)

    def reply(self, obj):
        """
        Serialize the provided object and send it as a reply to the client.

        :param obj: Any: The Python object to be serialized and sent
        """
        try:
            obj_str = json.dumps(obj, cls=self._plugin_encoder, **self._plugin_kwargs,
                                 serializer_kwrags=self._serializer_kwargs)
            obj_msg = std_msgs.msg.String()
            obj_msg.data = obj_str
            ROSNativeObjectServer.SEND_QUEUE.put(obj_msg, block=False)
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")


@Servers.register("Image", "ros")
class ROSImageServer(ROSServer):
    SEND_QUEUE = queue.Queue(maxsize=1)
    RECEIVE_QUEUE = queue.Queue(maxsize=1)

    def __init__(self, name: str, out_topic: str, carrier: str = "tcp",
                 width: int = -1, height: int = -1, rgb: bool = True, fp: bool = False,
                 deserializer_kwargs: Optional[dict] = None, **kwargs):
        """
        Specific server handling native Python objects, serializing them to JSON strings for transmission.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for REQ/REP pattern. Default is 'tcp'
        :param width: int: Width of the image. Default is -1 (use the width of the received image)
        :param height: int: Height of the image. Default is -1 (use the height of the received image)
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, out_topic, carrier=carrier, **kwargs)
        if "jpg" in kwargs:
            logging.warning("[ROS] ROS currently does not support JPG encoding in REQ/REP. Using raw image.")
            kwargs.pop("jpg")
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

        self._plugin_kwargs = kwargs
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._server = None

    def establish(self):
        """
        Establish the connection to the server.
        """
        self._server = rospy.Service(self.out_topic, ROSImageService, self._service_callback)
        self.established = True

    def await_request(self, *args, **kwargs):
        """
        Await and deserialize the client's request, returning the extracted arguments and keyword arguments.
        The method blocks until a message is received, then attempts to deserialize it using the configured JSON decoder
        hook, returning the extracted arguments and keyword arguments.

        :return: Tuple[list, dict]: A tuple containing two items:
                    - A list of arguments extracted from the received message
                    - A dictionary of keyword arguments extracted from the received message
        """
        if not self.established:
            self.establish()
        try:
            request = ROSImageServer.RECEIVE_QUEUE.get(block=True)
            [args, kwargs] = json.loads(request.data, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
            return args, kwargs
        except rospy.ServiceException as e:
            logging.error("[ROS] Service call failed: %s" % e)
            return [], {}

    @staticmethod
    def _service_callback(msg):
        """
        Callback for the ROS service.

        :param msg: ROSImageService._request_class: The request message
        :return: ROSImageService._response_class: The response message
        """
        ROSImageServer.RECEIVE_QUEUE.put(msg)
        return ROSImageServer.SEND_QUEUE.get(block=True)

    def reply(self, img: np.ndarray):
        """
        Serialize the provided image and send it as a reply to the client.

        :param img: np.ndarray: Image to publish
        """
        try:
            if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                    not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
                raise ValueError("Incorrect image shape for publisher")
            img = np.require(img, dtype=self._type, requirements='C')
            img_msg = sensor_msgs.msg.Image()
            img_msg.header.stamp = rospy.Time.now()
            img_msg.height = img.shape[0]
            img_msg.width = img.shape[1]
            img_msg.encoding = self._encoding
            img_msg.is_bigendian = img.dtype.byteorder == '>' or (img.dtype.byteorder == '=' and sys.byteorder == 'big')
            img_msg.step = img.strides[0]
            img_msg.data = img.tobytes()
            ROSImageServer.SEND_QUEUE.put(img_msg, block=False)
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")


@Servers.register("AudioChunk", "ros")
class ROSAudioChunkServer(ROSServer):
    SEND_QUEUE = queue.Queue(maxsize=1)
    RECEIVE_QUEUE = queue.Queue(maxsize=1)

    def __init__(self, name: str, out_topic: str, carrier: str = "tcp",
                 channels: int = 1, rate: int = 44100, chunk: int = -1,
                 deserializer_kwargs: Optional[dict] = None, **kwargs):
        """
        Specific server handling audio data as numpy arrays.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for REQ/REP pattern. Default is 'tcp'
        :param channels: int: Number of channels in the audio. Default is 1
        :param rate: int: Sampling rate of the audio. Default is 44100
        :param chunk: int: Number of samples in the audio chunk. Default is -1 (use the chunk size of the received audio)
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, out_topic, carrier=carrier, **kwargs)
        self._plugin_kwargs = kwargs
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._server = self._rep_msg = None

        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def establish(self):
        """
        Establish the connection to the server.
        """
        try:
            from wrapyfi_ros_interfaces.srv import ROSAudioService
        except ImportError:
            import wrapyfi
            logging.error("[ROS] Could not import ROSAudioService. "
                          "Make sure the ROS services in wrapyfi_extensions/wrapyfi_ros_interfaces are compiled. "
                          "Refer to the documentation for more information: \n" +
                          wrapyfi.__url__ + "wrapyfi_extensions/wrapyfi_ros_interfaces/README.md")
            sys.exit(1)
        self._server = rospy.Service(self.out_topic, ROSAudioService, self._service_callback)
        self._rep_msg = ROSAudioService._response_class().response
        self.established = True

    def await_request(self, *args, **kwargs):
        """
        Await and deserialize the client's request, returning the extracted arguments and keyword arguments.
        The method blocks until a message is received, then attempts to deserialize it using the configured JSON decoder
        hook, returning the extracted arguments and keyword arguments.

        :return: Tuple[list, dict]: A tuple containing two items:
                 - A list of arguments extracted from the received message
                 - A dictionary of keyword arguments extracted from the received message
        """
        if not self.established:
            self.establish()
        try:
            request = ROSAudioChunkServer.RECEIVE_QUEUE.get(block=True)
            [args, kwargs] = json.loads(request.request, object_hook=self._plugin_decoder_hook,
                                        **self._deserializer_kwargs)
            return args, kwargs
        except rospy.ServiceException as e:
            logging.error("[ROS] Service call failed: %s" % e)
            return [], {}

    @staticmethod
    def _service_callback(msg):
        """
        Callback for the ROS service.

        :param msg: ROSAudioService._request_class: The request message
        :return: ROSAudioService._response_class: The response message
        """
        ROSAudioChunkServer.RECEIVE_QUEUE.put(msg)
        return ROSAudioChunkServer.SEND_QUEUE.get(block=True)

    def reply(self, aud: Tuple[np.ndarray, int]):
        """
        Serialize the provided audio data and send it as a reply to the client.

        :param aud: Tuple[np.ndarray, int]: Audio chunk to publish formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        try:
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

            aud_msg = self._rep_msg
            aud_msg.header.stamp = rospy.Time.now()
            aud_msg.chunk_size = chunk
            aud_msg.channels = channels
            aud_msg.sample_rate = rate
            aud_msg.is_bigendian = aud.dtype.byteorder == '>' or (aud.dtype.byteorder == '=' and sys.byteorder == 'big')
            aud_msg.encoding = 'S16BE' if aud_msg.is_bigendian else 'S16LE'
            aud_msg.step = aud.strides[0]
            aud_msg.data = aud.tobytes()  # (aud * 32767.0).tobytes()
            ROSAudioChunkServer.SEND_QUEUE.put(aud_msg, block=False)
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
