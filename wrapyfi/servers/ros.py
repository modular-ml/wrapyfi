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

from wrapyfi.connect.servers import Server, Servers
from wrapyfi.middlewares.ros import ROSMiddleware, ROSNativeObjectService, ROSImageService
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class ROSServer(Server):

    def __init__(self, name, out_topic, carrier="", out_topic_connect=None, ros_kwargs=None, **kwargs):
        super().__init__(name, out_topic, carrier=carrier, out_topic_connect=out_topic_connect, **kwargs)
        ROSMiddleware.activate(**ros_kwargs or {})

    def close(self):
        if hasattr(self, "_server"):
            self._server.shutdown()

    def __del__(self):
        self.close()


@Servers.register("NativeObject", "ros")
class ROSNativeObjectServer(ROSServer):
    SEND_QUEUE = queue.Queue(maxsize=1)
    RECEIVE_QUEUE = queue.Queue(maxsize=1)

    def __init__(self, name, out_topic, carrier="", out_topic_connect=None, serializer_kwargs=None, deserializer_kwargs=None, **kwargs):
        super().__init__(name, out_topic, carrier=carrier, out_topic_connect=out_topic_connect, **kwargs)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._server = None

    def establish(self):
        self._server = rospy.Service(self.out_topic, ROSNativeObjectService, self._service_callback)
        self.established = True

    def await_request(self, *args, **kwargs):
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
       ROSNativeObjectServer.RECEIVE_QUEUE.put(msg)
       return ROSNativeObjectServer.SEND_QUEUE.get(block=True)

    def reply(self, obj):
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

    def __init__(self, name, out_topic, carrier="", out_topic_connect=None, width=-1, height=-1, rgb=True, fp=False, deserializer_kwargs=None, **kwargs):
        super().__init__(name, out_topic, carrier=carrier, out_topic_connect=out_topic_connect, **kwargs)
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
        self._server = rospy.Service(self.out_topic, ROSImageService, self._service_callback)
        self.established = True

    def await_request(self, *args, **kwargs):
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
       ROSImageServer.RECEIVE_QUEUE.put(msg)
       return ROSImageServer.SEND_QUEUE.get(block=True)

    def reply(self, img):
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

    def __init__(self, name, out_topic, carrier="", out_topic_connect=None, channels=1, rate=44100, chunk=-1,
                 deserializer_kwargs=None, **kwargs):
        super().__init__(name, out_topic, carrier=carrier, out_topic_connect=out_topic_connect, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._plugin_kwargs = kwargs
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._server = self._sound_msg = None

    def establish(self):
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
        self._sound_msg = ROSAudioService._response_class().response
        self.established = True

    def await_request(self, *args, **kwargs):
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
        ROSAudioChunkServer.RECEIVE_QUEUE.put(msg)
        return ROSAudioChunkServer.SEND_QUEUE.get(block=True)

    def reply(self, aud):
        try:
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
            ROSAudioChunkServer.SEND_QUEUE.put(aud_msg, block=False)
        except queue.Full:
            logging.warning(f"[ROS] Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
