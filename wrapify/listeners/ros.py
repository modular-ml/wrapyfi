import json
import queue
import numpy as np
import rospy
import std_msgs.msg
import sensor_msgs.msg

from wrapify.connect.listeners import Listener, ListenerWatchDog, Listeners
from wrapify.middlewares.ros import ROSMiddleware
from wrapify.utils import JsonDecodeHook


class ROSListener(Listener):

    def __init__(self, name, in_port, carrier="", should_wait=True, queue_size=5, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, **kwargs)
        ROSMiddleware.activate()
        self.queue_size = queue_size


@Listeners.register("NativeObject", "ros")
class ROSNativeObjectListener(ROSListener):

    def __init__(self, name, in_port, carrier="", should_wait=True, queue_size=5, load_torch_device=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, queue_size=queue_size, **kwargs)
        self._json_object_hook = JsonDecodeHook(torch_device=load_torch_device).object_hook
        self._subscriber = self._queue = None
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
            return json.loads(obj_str, object_hook=self._json_object_hook)
        except queue.Empty:
            return None

    def _message_callback(self, data):
        try:
            self._queue.put(data.data, block=False)
        except queue.Full:
            print(f"Discarding data because listener queue is full: {self.in_port}")


@Listeners.register("Image", "ros")
class ROSImageListener(ROSListener):

    def __init__(self, name, in_port, carrier="", should_wait=True, queue_size=5, width=-1, height=-1, rgb=True, fp=False, **kwargs):
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
            print(f"Discarding data because listener queue is full: {self.in_port}")


@Listeners.register("AudioChunk", "ros")
class ROSAudioChunkListener(ROSListener):

    def __init__(self, name, in_port, carrier="", should_wait=True, queue_size=5, channels=1, rate=44100, chunk=-1, **kwargs):
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
            print(f"Discarding data because listener queue is full: {self.in_port}")


@Listeners.register("Properties", "ros")
class ROSPropertiesListener(ROSListener):

    def __init__(self, name, in_port, **kwargs):
        super().__init__(name, in_port, **kwargs)
        raise NotImplementedError
