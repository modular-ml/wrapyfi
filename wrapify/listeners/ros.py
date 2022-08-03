import contextlib
import json
import queue
import rospy
import std_msgs.msg

from wrapify.connect.listeners import Listener, ListenerWatchDog, Listeners
from wrapify.middlewares.ros import ROSMiddleware
from wrapify.utils import JsonDecodeHook


class ROSListener(Listener):

    def __init__(self, name, in_port, carrier="", should_wait=False, queue_size=5):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)
        ROSMiddleware.activate()
        self.queue_size = queue_size


@Listeners.register("NativeObject", "ros")
class ROSNativeObjectListener(ROSListener):

    def __init__(self, name, in_port, carrier="", should_wait=False, queue_size=5, load_torch_device=None):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, queue_size=queue_size)
        self._json_object_hook = JsonDecodeHook(torch_device=load_torch_device).object_hook
        self._subscriber = None
        self._queue = None
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

    def __init__(self, name, in_port, carrier="", should_wait=False, width=320, height=240, rgb=True):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)
        self.width = width
        self.height = height
        self.rgb = rgb
        ListenerWatchDog().add_listener(self)
        raise NotImplementedError


@Listeners.register("AudioChunk", "ros")
class ROSAudioChunkListener(ROSListener):

    def __init__(self, name, in_port, carrier="", should_wait=False, channels=1, rate=44100, chunk=-1):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        ListenerWatchDog().add_listener(self)
        raise NotImplementedError


@Listeners.register("Properties", "ros")
class ROSPropertiesListener(ROSListener):

    def __init__(self, name, in_port, **kwargs):
        super().__init__(name, in_port, **kwargs)
        raise NotImplementedError
