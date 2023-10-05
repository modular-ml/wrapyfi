import logging
import sys
import json
import time
import os
import importlib.util
import queue

import numpy as np
import rospy
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.clients import Client, Clients
from wrapyfi.middlewares.ros import ROSMiddleware, ROSNativeObjectService, ROSImageService
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class ROSClient(Client):

    def __init__(self, name, in_topic, carrier="", ros_kwargs=None, **kwargs):
        super().__init__(name, in_topic, carrier=carrier, **kwargs)
        ROSMiddleware.activate(**ros_kwargs or {})

    def close(self):
        if hasattr(self, "_client"):
            if self._client is not None:
                self._client.shutdown()

    def __del__(self):
        self.close()


@Clients.register("NativeObject", "ros")
class ROSNativeObjectClient(ROSClient):

    def __init__(self, name, in_topic, carrier="", serializer_kwargs=None, deserializer_kwargs=None, persistent=False, **kwargs):
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
        rospy.wait_for_service(self.in_topic)
        self._client = rospy.ServiceProxy(self.in_topic, ROSNativeObjectService, persistent=self.persistent)
        if self.persistent:
            self.established = True

    def request(self, *args, **kwargs):
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except rospy.ServiceException as e:
            logging.error("[ROS] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        # transmit args to server
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = std_msgs.msg.String()
        args_msg.data = args_str
        # receive message from server
        msg = self._client(args_msg)
        obj = json.loads(msg.data, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        self._queue.put(obj, block=False)

    def _await_reply(self):
        try:
            reply = self._queue.get(block=True)
            return reply
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
            return None

@Clients.register("Image", "ros")
class ROSImageClient(ROSClient):

    def __init__(self, name, in_topic, carrier="", width=-1, height=-1, rgb=True, fp=False, serializer_kwargs=None, persistent=False, **kwargs):
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
        rospy.wait_for_service(self.in_topic)
        self._client = rospy.ServiceProxy(self.in_topic, ROSImageService, persistent=self.persistent)
        if self.persistent:
            self.established = True

    def request(self, *args, **kwargs):
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except rospy.ServiceException as e:
            logging.error("[ROS] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        # transmit args to server
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = std_msgs.msg.String()
        args_msg.data = args_str
        # receive message from server
        msg = self._client(args_msg)
        self._queue.put((msg.height, msg.width, msg.encoding, msg.is_bigendian, msg.data), block=False)

    def _await_reply(self):
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

    def __init__(self, name, in_topic, carrier="", channels=1, rate=44100, chunk=-1, serializer_kwargs=None, persistent=False, **kwargs):
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
        rospy.wait_for_service(self.in_topic)
        self._client = rospy.ServiceProxy(self.in_topic, ROSImageService, persistent=self.persistent)
        if self.persistent:
            self.established = True

    def request(self, *args, **kwargs):
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except rospy.ServiceException as e:
            logging.error("[ROS] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        # transmit args to server
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = std_msgs.msg.String()
        args_msg.data = args_str
        # receive message from server
        msg = self._client(args_msg)
        self._queue.put((msg.height, msg.width, msg.encoding, msg.is_bigendian, msg.data), block=False)

    def _await_reply(self):
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

