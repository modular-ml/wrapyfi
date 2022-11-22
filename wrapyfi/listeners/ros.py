import json
import logging
import queue
import time
import os

import numpy as np
import rospy
import rostopic
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.listeners import Listener, ListenerWatchDog, Listeners
from wrapyfi.middlewares.ros import ROSMiddleware
from wrapyfi.encoders import JsonDecodeHook


QUEUE_SIZE = int(os.environ.get("WRAPYFI_ROS_QUEUE_SIZE", 5))
WATCHDOG_POLL_REPEAT = None


class ROSListener(Listener):

    def __init__(self, name, in_port, carrier="", should_wait=True, queue_size=QUEUE_SIZE, ros_kwargs=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, **kwargs)
        ROSMiddleware.activate(**ros_kwargs or {})
        self.queue_size = queue_size

    def close(self):
        if hasattr(self, "_subscriber"):
            self._subscriber.shutdown()

    def __del__(self):
        self.close()


@Listeners.register("NativeObject", "ros")
class ROSNativeObjectListener(ROSListener):

    def __init__(self, name, in_port, carrier="", should_wait=True, queue_size=QUEUE_SIZE, deserializer_kwargs=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)
        self._subscriber = self._queue = None

        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        ListenerWatchDog().add_listener(self)

    def establish(self):
        self._queue = queue.Queue(maxsize=0 if self.queue_size is None or self.queue_size <= 0 else self.queue_size)
        self._subscriber = rospy.Subscriber(self.in_port, std_msgs.msg.String, callback=self._message_callback)
        self.established = True

    def listen(self):
        if not self.established:
            self.establish()
        try:
            obj_str = self._queue.get(block=self.should_wait)
            return json.loads(obj_str, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        except queue.Empty:
            return None

    def _message_callback(self, msg):
        try:
            self._queue.put(msg.data, block=False)
        except queue.Full:
            logging.warning(f"Discarding data because listener queue is full: {self.in_port}")


@Listeners.register("Image", "ros")
class ROSImageListener(ROSListener):

    def __init__(self, name, in_port, carrier="", should_wait=True, queue_size=QUEUE_SIZE, width=-1, height=-1, rgb=True, fp=False, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)
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
        self._subscriber = self._queue = None
        ListenerWatchDog().add_listener(self)

    def establish(self):
        self._queue = queue.Queue(maxsize=0 if self.queue_size is None or self.queue_size <= 0 else self.queue_size)
        self._subscriber = rospy.Subscriber(self.in_port, sensor_msgs.msg.Image, callback=self._message_callback)
        self.established = True

    def listen(self):
        if not self.established:
            self.establish()
        try:
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
        try:
            self._queue.put((data.height, data.width, data.encoding, data.is_bigendian, data.data), block=False)
        except queue.Full:
            logging.warning(f"Discarding data because listener queue is full: {self.in_port}")


@Listeners.register("AudioChunk", "ros")
class ROSAudioChunkListener(ROSListener):

    def __init__(self, name, in_port, carrier="", should_wait=True, queue_size=QUEUE_SIZE, channels=1, rate=44100, chunk=-1, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self._subscriber = self._queue = None
        ListenerWatchDog().add_listener(self)

    def establish(self):
        self._queue = queue.Queue(maxsize=0 if self.queue_size is None or self.queue_size <= 0 else self.queue_size)
        self._subscriber = rospy.Subscriber(self.in_port, sensor_msgs.msg.Image, callback=self._message_callback)
        self.established = True

    def listen(self):
        if not self.established:
            self.establish()
        try:
            chunk, channels, encoding, is_bigendian, data = self._queue.get(block=self.should_wait)
            if encoding != '32FC1':
                raise ValueError("Incorrect encoding for listener")
            elif 0 < self.chunk != chunk or self.channels != channels or len(data) != chunk * channels * 4:
                raise ValueError("Incorrect audio shape for listener")
            aud = np.frombuffer(data, dtype=np.dtype(np.float32).newbyteorder('>' if is_bigendian else '<')).reshape((chunk, channels))
            return aud, self.rate
        except queue.Empty:
            return None, self.rate

    def _message_callback(self, data):
        try:
            self._queue.put((data.height, data.width, data.encoding, data.is_bigendian, data.data), block=False)
        except queue.Full:
            logging.warning(f"Discarding data because listener queue is full: {self.in_port}")


@Listeners.register("Properties", "ros")
class ROSPropertiesListener(ROSListener):

    def __init__(self, name, in_port, carrier="", should_wait=True, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, **kwargs)
        self._subscriber = self._queue = None

        if not self.should_wait:
            ListenerWatchDog().add_listener(self)

        self.previous_property = False

    def await_connection(self, port=None, repeats=None):
        connected = False
        if port is None:
            port = self.in_port
        logging.info(f"Waiting for property: {port}")
        if repeats is None:
            if self.should_wait:
                repeats = -1
            else:
                repeats = 1

            while repeats > 0 or repeats <= -1:
                repeats -= 1
                self.previous_property = rospy.get_param(self.in_port, False)
                connected = True if self.previous_property else False
                if connected:
                    logging.info(f"Found property: {port}")
                    break
                time.sleep(0.2)
        return connected

    def establish(self, repeats=None, **kwargs):
        established = self.await_connection(repeats=repeats)
        return self.check_establishment(established)

    def listen(self):
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                obj = None
            else:
                obj = self.previous_property
            return obj
        else:
            obj = rospy.get_param(self.in_port, False)
            return obj


@Listeners.register("ROSMessage", "ros")
class ROSMessageListener(ROSListener):

    def __init__(self, name, in_port, carrier="", should_wait=True, queue_size=QUEUE_SIZE, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)
        self._subscriber = self._queue = None
        self._topic_type = None
        ListenerWatchDog().add_listener(self)

    def establish(self):
        self._queue = queue.Queue(maxsize=0 if self.queue_size is None or self.queue_size <= 0 else self.queue_size)
        self._topic_type, topic_str, _ = rostopic.get_topic_class(self.in_port, blocking=self.should_wait)
        if self._topic_type is None:
            return
        self._subscriber = rospy.Subscriber(self.in_port, self._topic_type, callback=self._message_callback)
        self.established = True

    def listen(self):
        if not self.established:
            self.establish()
        try:
            obj = self._queue.get(block=self.should_wait)

            return obj  # self._topic_type.deserialize_numpy(obj_str)
        except queue.Empty:
            return None

    def _message_callback(self, msg):
        try:
            self._queue.put(msg, block=False)
        except queue.Full:
            logging.warning(f"Discarding data because listener queue is full: {self.in_port}")

