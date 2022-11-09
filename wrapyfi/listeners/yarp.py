import logging
import json
import time

import numpy as np
import yarp

from wrapyfi.connect.listeners import Listener, ListenerWatchDog, Listeners
from wrapyfi.middlewares.yarp import YarpMiddleware
from wrapyfi.encoders import JsonDecodeHook


class YarpListener(Listener):

    def __init__(self, name, in_port, carrier="", yarp_kwargs=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        YarpMiddleware.activate(**yarp_kwargs or {})

    def await_connection(self, port=None, repeats=None):
        connected = False
        if port is None:
            port = self.in_port
        logging.info(f"Waiting for input port: {port}")
        if repeats is None:
            if self.should_wait:
                repeats = -1
            else:
                repeats = 1

            while repeats > 0 or repeats <= -1:
                repeats -= 1
                connected = yarp.Network.exists(port)
                if connected:
                    logging.info(f"Connected to input port: {port}")
                    break
                time.sleep(0.2)
        return connected

    def read_port(self, port):
        while True:
            obj = port.read(shouldWait=False)
            if self.should_wait and obj is None:
                time.sleep(0.005)
            else:
                return obj

    def close(self):
        if hasattr(self, "_port") and self._port:
            self._port.close()

    def __del__(self):
        self.close()

@Listeners.register("NativeObject", "yarp")
class YarpNativeObjectListener(YarpListener):

    def __init__(self, name, in_port, carrier="", deserializer_kwargs=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        self._port = self._netconnect = None

        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self.deserializer_kwargs = deserializer_kwargs or {}

        if not self.should_wait:
            ListenerWatchDog().add_listener(self)

    def establish(self, repeats=None, **kwargs):
        established = self.await_connection(repeats=repeats)
        if established:
            self._port = yarp.BufferedPortBottle()
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._port.open(self.in_port + ":in" + rnd_id)
            self._netconnect = yarp.Network.connect(self.in_port, self.in_port + ":in" + rnd_id, self.carrier)
        return self.check_establishment(established)

    def listen(self):
        if not self.established:
            established = self.establish()
            if not established:
                return None
        obj = self.read_port(self._port)
        if obj is not None:
            return json.loads(obj.get(0).asString(), object_hook=self._plugin_decoder_hook, **self.deserializer_kwargs)
        else:
            return None



@Listeners.register("Image", "yarp")
class YarpImageListener(YarpListener):

    def __init__(self, name, in_port, carrier="", width=-1, height=-1, rgb=True, fp=False, **kwargs):
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self._port = self._type = self._netconnect = None
        if not self.should_wait:
            ListenerWatchDog().add_listener(self)

    def establish(self, repeats=None, **kwargs):
        established = self.await_connection(repeats=repeats)
        if established:
            if self.rgb:
                self._port = yarp.BufferedPortImageRgbFloat() if self.fp else yarp.BufferedPortImageRgb()
            else:
                self._port = yarp.BufferedPortImageFloat() if self.fp else yarp.BufferedPortImageMono()
            self._type = np.float32 if self.fp else np.uint8
            in_port_connect = f"{self.in_port}:in{np.random.randint(100000, size=1).item()}"
            self._port.open(in_port_connect)
            self._netconnect = yarp.Network.connect(self.in_port, in_port_connect, self.carrier)
        return self.check_establishment(established)

    def listen(self):
        if not self.established:
            established = self.establish()
            if not established:
                return None
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
        wrapper_img.setExternal(img.data, img.shape[1], img.shape[0])
        wrapper_img.copy(yarp_img)
        return img


@Listeners.register("AudioChunk", "yarp")
class YarpAudioChunkListener(YarpImageListener):

    def __init__(self, name, in_port, carrier="", channels=1, rate=44100, chunk=-1, **kwargs):
        super().__init__(name, in_port, carrier=carrier, width=chunk, height=channels, rgb=False, fp=True, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self._dummy_sound = self._dummy_port = self._dummy_netconnect = None
        if not self.should_wait:
            ListenerWatchDog().add_listener(self)

    def establish(self, repeats=None, **kwargs):
        established = self.await_connection(port=self.in_port + "_SND", repeats=repeats)
        if established:
            # create a dummy sound object for transmitting the sound props. This could be cleaner but left for future impl.
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._dummy_port = yarp.Port()
            self._dummy_port.open(self.in_port + "_SND:in" + rnd_id)
            self._dummy_netconnect = yarp.Network.connect(self.in_port + "_SND", self.in_port + "_SND:in" + rnd_id, self.carrier)
        established = self.check_establishment(established)
        established_parent = super(YarpAudioChunkListener, self).establish(repeats=repeats)
        if established_parent:
            self._dummy_sound = yarp.Sound()
            # self._dummy_port.read(self._dummy_sound)
            # self.rate = self._dummy_sound.getFrequency()
            # self.width = self.chunk = self._dummy_sound.getSamples()
            # self.height = self.channels = self._dummy_sound.getChannels()
        return established

    def listen(self):
        return super().listen(), self.rate

    def close(self):
        super().close()
        if self._dummy_port:
            self._dummy_port.close()


@Listeners.register("Properties", "yarp")
class YarpPropertiesListener(YarpListener):
    def __init__(self, name, in_port, **kwargs):
        super().__init__(name, in_port, **kwargs)
        raise NotImplementedError
