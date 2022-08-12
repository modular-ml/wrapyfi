import json
import time
import numpy as np
import yarp

from wrapify.connect.listeners import Listener, ListenerWatchDog, Listeners
from wrapify.middlewares.yarp import YarpMiddleware
from wrapify.utils import JsonDecodeHook


class YarpListener(Listener):

    def __init__(self, name, in_port, carrier="", should_wait=True, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, **kwargs)
        YarpMiddleware.activate()

    def await_connection(self, port=None):
        if port is None:
            port = self.in_port
        print("Waiting for input port:", port)
        while not yarp.Network.exists(port):
            time.sleep(0.2)
        print("Connected to input port:", port)

    def read_port(self, port):
        while True:
            obj = port.read(shouldWait=False)
            if self.should_wait and obj is None:
                time.sleep(0.005)
            else:
                return obj


@Listeners.register("NativeObject", "yarp")
class YarpNativeObjectListener(YarpListener):

    def __init__(self, name, in_port, carrier="", should_wait=True, load_torch_device=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, **kwargs)
        self._json_object_hook = JsonDecodeHook(torch_device=load_torch_device).object_hook
        self._port = self._netconnect = None
        ListenerWatchDog().add_listener(self)

    def establish(self):
        print("Waiting for input port:", self.in_port)
        while not yarp.Network.exists(self.in_port):
            time.sleep(0.2)
        print("Connected to input port:", self.in_port)
        self._port = yarp.BufferedPortBottle()
        rnd_id = str(np.random.randint(100000, size=1)[0])
        self._port.open(self.in_port + ":in" + rnd_id)
        self._netconnect = yarp.Network.connect(self.in_port, self.in_port + ":in" + rnd_id, self.carrier)
        self.established = True

    def listen(self):
        if not self.established:
            self.establish()
        obj = self.read_port(self._port)
        return json.loads(obj.get(0).asString(), object_hook=self._json_object_hook) if obj is not None else None


@Listeners.register("Image", "yarp")
class YarpImageListener(YarpListener):

    def __init__(self, name, in_port, carrier="", should_wait=True, width=-1, height=-1, rgb=True, fp=False, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self._port = self._type = self._netconnect = None
        ListenerWatchDog().add_listener(self)

    def establish(self):
        self.await_connection()
        if self.rgb:
            self._port = yarp.BufferedPortImageRgbFloat() if self.fp else yarp.BufferedPortImageRgb()
        else:
            self._port = yarp.BufferedPortImageFloat() if self.fp else yarp.BufferedPortImageMono()
        self._type = np.float32 if self.fp else np.uint8
        in_port_connect = f"{self.in_port}:in{np.random.randint(100000, size=1).item()}"
        self._port.open(in_port_connect)
        self._netconnect = yarp.Network.connect(self.in_port, in_port_connect, self.carrier)
        self.established = True

    def listen(self):
        if not self.established:
            self.establish()
        yarp_img = self.read_port(self._port)
        if yarp_img is None:
            return None
        elif 0 < self.width != yarp_img.width() or 0 < self.height != yarp_img.height():
            raise ValueError("Incorrect image shape for listener")
        if self.rgb:
            img = np.zeros((yarp_img.height(), yarp_img.width(), 3), dtype=self._type, order='C')
            wrapper_img = yarp.ImageRgbFloat() if self.fp else yarp.ImageRgb()
        else:
            img = np.zeros((yarp_img.height(), yarp_img.width()), dtype=self._type, order='C')
            wrapper_img = yarp.ImageFloat() if self.fp else yarp.ImageMono()
        wrapper_img.resize(img.shape[1], img.shape[0])
        wrapper_img.setExternal(img, img.shape[1], img.shape[0])
        wrapper_img.copy(yarp_img)
        return img

    def close(self):
        if self._port:
            self._port.close()

    def __del__(self):
        self.close()


@Listeners.register("AudioChunk", "yarp")
class YarpAudioChunkListener(YarpImageListener):
    def __init__(self, name, in_port, carrier="", should_wait=True, channels=1, rate=44100, chunk=-1, **kwargs):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait, width=chunk, height=channels, rgb=False, fp=True, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self._dummy_sound = self._dummy_port = self._dummy_netconnect = None
        ListenerWatchDog().add_listener(self)

    def establish(self):
        self.await_connection(port=self.in_port + "_SND")
        if self.channels == -1 or self.rate == -1 or self.chunk == -1:
            # create a dummy sound object for transmitting the sound props. This could be cleaner but left for future impl.
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._dummy_port = yarp.Port()
            self._dummy_port.open(self.in_port + "_SND:in" + rnd_id)
            self._dummy_netconnect = yarp.Network.connect(self.in_port + "_SND", self.in_port + "_SND:in" + rnd_id, self.carrier)
            self._dummy_sound = yarp.Sound()
            self.rate = self._dummy_sound.getFrequency()
            self.chunk = self._dummy_sound.getSamples()
            self.channels = self._dummy_sound.getChannels()
            self.width = self.chunk
            self.height = self.channels
        super(YarpAudioChunkListener, self).establish()

    def listen(self):
        if not self.established:
            self.establish()
        return self._listener(), self.rate


@Listeners.register("Properties", "yarp")
class YarpPropertiesListener(YarpListener):

    def __init__(self, name, in_port, **kwargs):
        super().__init__(name, in_port, **kwargs)
        raise NotImplementedError
