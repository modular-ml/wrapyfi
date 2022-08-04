import json
import time
import numpy as np
import yarp

from wrapify.connect.listeners import Listener, ListenerWatchDog, Listeners
from wrapify.middlewares.yarp import YarpMiddleware
from wrapify.utils import JsonDecodeHook


class YarpListener(Listener):

    def __init__(self, name, in_port, carrier="", should_wait=True):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)
        YarpMiddleware.activate()


@Listeners.register("Image", "yarp")
class YarpImageListener(YarpListener):
    def __init__(self, name, in_port, carrier="", should_wait=True, width=320, height=240, rgb=True):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)
        self.width = width
        self.height = height
        self.rgb = rgb
        self._port = self._iarray = self._netconnect = self._type = self._listener = None
        ListenerWatchDog().add_listener(self)

    def establish(self):
        print("Waiting for input port:", self.in_port)
        while not yarp.Network.exists(self.in_port):
            time.sleep(0.2)
        print("Connected to input port:", self.in_port)
        if self.rgb:
            self._port = yarp.BufferedPortImageRgb()
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._port.open(self.in_port + ":in" + rnd_id)
            self._netconnect = yarp.Network.connect(self.in_port, self.in_port + ":in" + rnd_id, self.carrier)
            if self.height != -1 and self.width != -1:
                self._iarray = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self._type = np.uint8
        else:
            self._port = yarp.BufferedPortImageFloat()
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._port.open(self.in_port + ":in" + rnd_id)
            self._netconnect = yarp.Network.connect(self.in_port, self.in_port + ":in" + rnd_id, self.carrier)
            if self.height != -1 and self.width != -1:
                self._iarray = np.zeros((self.height, self.width), dtype=np.float32)
            self._type = np.float32

        # set the listener
        if self.width == -1 and self.height == -1:
            self._listener = self._listen_unk_width_height
        elif self.width == -1:
            self._listener = self._listen_unk_width
        elif self.height == -1:
            self._listener = self._listen_unk_height
        else:
            self._listener = self._listen

        self.established = True

    def _listen_unk_width(self):
        # TODO (fabawi): dynamic width only
        img = self._port.read(shouldWait=self.should_wait)
        if img is not None:
            self._iarray = np.zeros((img.height(), img.width()), dtype=self._type)
            img.setExternal(self._iarray.data,
                            img.width(), img.height())
        return self._iarray

    def _listen_unk_height(self):
        # TODO (fabawi): dynamic height only
        img = self._port.read(shouldWait=self.should_wait)
        if img is not None:
            self._iarray = np.zeros((img.height(), img.width()), dtype=self._type)
            img.setExternal(self._iarray.data,
                            img.width(), img.height())
        return self._iarray

    def _listen_unk_width_height(self):
        img = self._port.read(shouldWait=self.should_wait)
        if img is not None:
            self._iarray = np.zeros((img.height(), img.width()), dtype=self._type)
            img.setExternal(self._iarray.data,
                            img.width(), img.height())
        return self._iarray

    def _listen(self):
        img = self._port.read(shouldWait=self.should_wait)
        if img is not None:
            img.setExternal(self._iarray.data, self._iarray.shape[1], self._iarray.shape[0])
        return self._iarray

    def listen(self):
        if not self.established:
            self.establish()
        return self._listener()

    def close(self):
        if self._port:
            self._port.close()

    def __del__(self):
        self.close()


@Listeners.register("AudioChunk", "yarp")
class YarpAudioChunkListener(YarpImageListener):
    def __init__(self, name, in_port, carrier="", should_wait=True, channels=1, rate=44100, chunk=-1):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait,
                         width=chunk, height=channels, rgb=False)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._dummy_sound = self._dummy_port = self._dummy_netconnect = None
        ListenerWatchDog().add_listener(self)

    def establish(self):
        print("Waiting for input port:", self.in_port + "_SND")
        while not yarp.Network.exists(self.in_port + "_SND"):
            time.sleep(0.2)
        print("Connected to input port:", self.in_port + "_SND")

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

            super(YarpAudioChunkListener, self).width = self.chunk
            super(YarpAudioChunkListener, self).height = self.channels

        super(YarpAudioChunkListener, self).establish()

    def listen(self):
        if not self.established:
            self.establish()
        return self._listener(), self.rate


@Listeners.register("NativeObject", "yarp")
class YarpNativeObjectListener(YarpListener):
    def __init__(self, name, in_port, carrier="", should_wait=True, load_torch_device=None):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)

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
        while True:
            obj = self._port.read(shouldWait=False)
            if self.should_wait and obj is None:
                time.sleep(0.005)
            else:
                break
        return json.loads(obj.get(0).asString(), object_hook=self._json_object_hook) if obj is not None else None


@Listeners.register("Properties", "yarp")
class YarpPropertiesListener(YarpListener):
    def __init__(self, name, in_port, **kwargs):
        super().__init__(name, in_port, **kwargs)
        raise NotImplementedError
