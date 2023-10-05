import logging
import sys
import json
import time
import os
import importlib.util
import queue
from typing import Optional, Any

import numpy as np
import rospy
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.clients import Client, Clients
from wrapyfi.middlewares.ros import ROSMiddleware, ROSNativeObjectService, ROSImageService
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class ROSClient(Client):
    def __init__(self, name: str, in_topic: str, carrier: str = "tcp",
                 ros_kwargs: Optional[dict] = None, **kwargs):
        """
        Initialize the client

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for rep/req pattern. Default is 'tcp'
        :param ros_kwargs: dict, optional: Additional kwargs for the ROS middleware. Defaults to None
        :param kwargs: dict: Additional kwargs for the Client
        """
        if carrier != "tcp":
            logging.warning("[ROS] ROS does not support other carriers than TCP for req/rep pattern. Using TCP.")
            carrier = "tcp"
        super().__init__(name, in_topic, carrier=carrier, **kwargs)
        ROSMiddleware.activate(**ros_kwargs or {})

    def close(self):
        """
        Close the client
        """
        if hasattr(self, "_client") and self._client:
            self._client.shutdown()

    def __del__(self):
        self.close()


@Clients.register("NativeObject", "ros")
class ROSNativeObjectClient(ROSClient):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp",
                 serializer_kwargs: Optional[dict] = None,
                 deserializer_kwargs: Optional[dict] = None,
                 persistent: bool = False, **kwargs):
        """
        The NativeObject client using the ROS String message assuming the data is serialized as a JSON string.
        Deserializes the data (including plugins) using the decoder and parses it to a Python object

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for rep/req pattern. Default is 'tcp'
        :param serializer_kwargs: dict, optional: Additional kwargs for the serializer. Defaults to None
        :param deserializer_kwargs: dict, optional: Additional kwargs for the deserializer. Defaults to None
        :param persistent: bool: Whether to keep the service connection alive across multiple service calls. Default is False
        :param kwargs: dict: Additional kwargs
        """
        super().__init__(name, in_topic, carrier=carrier, **kwargs)
        self._client = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self.persistent = persistent

    def establish(self):
        """
        Establish the client's connection to the ROS service
        """
        rospy.wait_for_service(self.in_topic)
        self._client = rospy.ServiceProxy(self.in_topic, ROSNativeObjectService, persistent=self.persistent)
        if self.persistent:
            self.established = True

    def request(self, *args, **kwargs) -> Any:
        """
        Send a request to the ROS service

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        :return: Any: The response from the ROS service
        """
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except rospy.ServiceException as e:
            logging.error("[ROS] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        """
        Internal method to send a request to the ROS service

        :param args: tuple: Positional arguments to send in the request.
        :param kwargs: dict: Keyword arguments to send in the request.
        """
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = std_msgs.msg.String()
        args_msg.data = args_str

        msg = self._client(args_msg)
        obj = json.loads(msg.data, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        self._queue.put(obj, block=False)

    def _await_reply(self) -> Any:
        """
        Wait for and return the reply from the ROS service

        :return: Any: The response from the ROS service
        """
        try:
            reply = self._queue.get(block=True)
            return reply
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__class__.__name__}")
            return None


@Clients.register("Image", "ros")
class ROSImageClient(ROSClient):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", width: int = -1, height: int = -1,
                 rgb: bool = True, fp: bool = False, serializer_kwargs: Optional[dict] = None,
                 persistent: bool = False, **kwargs):
        """
        The Image client using the ROS Image message parsed to a numpy array

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for rep/req pattern. Default is 'tcp'
        :param width: int: The width of the image. Default is -1
        :param height: int: The height of the image. Default is -1
        :param rgb: bool: Whether the image is RGB. Default is True
        :param fp: bool: Whether to utilize floating-point precision. Default is False
        :param serializer_kwargs: dict, optional: Additional kwargs for the serializer. Defaults to None
        :param persistent: bool: Whether to keep the service connection alive across multiple service calls. Default is False
        :param kwargs: dict: Additional kwargs.
        """
        super().__init__(name, in_topic, carrier=carrier, **kwargs)
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
        self._pixel_bytes = (3 if self.rgb else 1) * np.dtype(self._type).itemsize

        self._client = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}

        self.persistent = persistent

    def establish(self):
        """
        Establish the client's connection to the ROS service
        """
        rospy.wait_for_service(self.in_topic)
        self._client = rospy.ServiceProxy(self.in_topic, ROSImageService, persistent=self.persistent)
        if self.persistent:
            self.established = True

    def request(self, *args, **kwargs):
        """
        Send a request to the ROS service

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        :return: np.array: The received image from the ROS service
        """
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except rospy.ServiceException as e:
            logging.error("[ROS] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        """
        Internal method to send a request to the ROS service

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        """
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = std_msgs.msg.String()
        args_msg.data = args_str
        msg = self._client(args_msg)
        self._queue.put((msg.height, msg.width, msg.encoding, msg.is_bigendian, msg.data), block=False)

    def _await_reply(self):
        """
        Wait for and return the reply from the ROS service

        :return: np.array: The received image from the ROS service
        """
        try:
            height, width, encoding, is_bigendian, data = self._queue.get(block=True)
            if encoding != self._encoding:
                raise ValueError("Incorrect encoding for listener")
            elif 0 < self.width != width or 0 < self.height != height or len(data) != height * width * self._pixel_bytes:
                raise ValueError("Incorrect image shape for listener")
            img = np.frombuffer(data, dtype=np.dtype(self._type).newbyteorder('>' if is_bigendian else '<')).reshape((height, width, -1))
            if img.shape[2] == 1:
                img = img.squeeze(axis=2)
            return img
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
            return None


@Clients.register("AudioChunk", "ros")
class ROSAudioChunkClient(ROSClient):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", channels: int = 1, 
                 rate: int = 44100, chunk: int = -1, serializer_kwargs: Optional[dict] = None, 
                 persistent: bool = False, **kwargs):
        """
        The AudioChunk client using the ROS Image message parsed to a numpy array

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ROS currently only supports TCP for rep/req pattern. Default is 'tcp'
        :param channels: int: Number of audio channels. Default is 1
        :param rate: int: Sampling rate of the audio. Default is 44100
        :param chunk: int: The size of audio chunks. Default is -1
        :param serializer_kwargs: dict, optional: Additional kwargs for the serializer. Defaults to None
        :param persistent: bool: Whether to keep the service connection alive across multiple service calls. Default is False
        :param kwargs: dict: Additional kwargs.
        """
        super().__init__(name, in_topic, carrier=carrier, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._client = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}

        self.persistent = persistent

    def establish(self):
        """
        Establish the client's connection to the ROS service
        """
        rospy.wait_for_service(self.in_topic)
        self._client = rospy.ServiceProxy(self.in_topic, ROSImageService, persistent=self.persistent)
        if self.persistent:
            self.established = True

    def request(self, *args, **kwargs):
        """
        Send a request to the ROS service

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        :return: Tuple[np.array, int]: The received audio chunk and rate from the ROS service
        """
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except rospy.ServiceException as e:
            logging.error("[ROS] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        """
        Internal method to send a request to the ROS service

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        """
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = std_msgs.msg.String()
        args_msg.data = args_str
        msg = self._client(args_msg)
        self._queue.put((msg.height, msg.width, msg.encoding, msg.is_bigendian, msg.data), block=False)

    def _await_reply(self):
        """
        Wait for and return the reply from the ROS service

        :return: Tuple[np.array, int]: The received audio chunk and rate from the ROS service
        """
        try:
            chunk, channels, encoding, is_bigendian, data = self._queue.get(block=self.should_wait)
            if encoding != '32FC1':
                raise ValueError("Incorrect encoding for listener")
            elif 0 < self.chunk != chunk or self.channels != channels or len(data) != chunk * channels * 4:
                raise ValueError("Incorrect audio shape for listener")
            aud = np.frombuffer(data, dtype=np.dtype(np.float32).newbyteorder('>' if is_bigendian else '<')).reshape(
                (chunk, channels))
            return aud, self.rate
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
            return None, self.rate

