import json
import logging

import numpy as np
try:
    import yarp
    yarp.Network.init()
    logging.info("YARP is found and can be used")
except:
    # print("Install YARP to use wrapify")
    logging.warning("YARP is not found and cannot be used")
    pass

from wrapify.connect.listeners import Listener, ListenerWatchDog, Listeners
import wrapify.utils


@Listeners.register("Image", "yarp")
class YarpImageListener(Listener):
    def __init__(self, name, in_port, carrier="", should_wait=False, width=320, height=240, rgb=True):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)
        self.width = width
        self.height = height
        self.rgb = rgb

        self._port, self._iarray, self.__netconnect, self.__type, self.__listener = [None] * 5
        ListenerWatchDog().add_listener(self)

    def establish(self):
        print("waiting for in_port: ", self.in_port)
        while not yarp.Network.exists(self.in_port):
            yarp.Network.waitPort(self.in_port, quiet=True)
        print("connected to in_port: ", self.in_port)
        if self.rgb:
            self._port = yarp.BufferedPortImageRgb()
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._port.open(self.in_port + ":in" + rnd_id)
            self.__netconnect = yarp.Network.connect(self.in_port, self.in_port + ":in" + rnd_id, self.carrier)
            if self.height != -1 and self.width != -1:
                self._iarray = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.__type = np.uint8
        else:
            self._port = yarp.BufferedPortImageFloat()
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._port.open(self.in_port + ":in" + rnd_id)
            self.__netconnect = yarp.Network.connect(self.in_port, self.in_port + ":in" + rnd_id, self.carrier)
            if self.height != -1 and self.width != -1:
                self._iarray = np.zeros((self.height, self.width), dtype=np.float32)
            self.__type = np.float32

        # set the listener
        if self.width == -1 and self.height == -1:
            self.__listener = self.__listen_unk_width_height
        elif self.width == -1:
            self.__listener = self.__listen_unk_width
        elif self.height == -1:
            self.__listener = self.__listen_unk_height
        else:
            self.__listener = self.__listen

        self.established = True

    def __listen_unk_width(self):
        # TODO (fabawi): dynamic width only
        img = self._port.read(shouldWait=self.should_wait)
        if img is not None:
            self._iarray = np.zeros((img.height(), img.width()), dtype=self.__type)
            img.setExternal(self._iarray.data,
                            img.width(), img.height())
        return self._iarray

    def __listen_unk_height(self):
        # TODO (fabawi): dynamic height only
        img = self._port.read(shouldWait=self.should_wait)
        if img is not None:
            self._iarray = np.zeros((img.height(), img.width()), dtype=self.__type)
            img.setExternal(self._iarray.data,
                            img.width(), img.height())
        return self._iarray

    def __listen_unk_width_height(self):
        img = self._port.read(shouldWait=self.should_wait)
        if img is not None:
            self._iarray = np.zeros((img.height(), img.width()), dtype=self.__type)
            img.setExternal(self._iarray.data,
                            img.width(), img.height())
        return self._iarray

    def __listen(self):
        img = self._port.read(shouldWait=self.should_wait)
        if img is not None:
            img.setExternal(self._iarray.data,
                        self._iarray.shape[1], self._iarray.shape[0])
        return self._iarray

    def listen(self):
        if not self.established:
            self.establish()
        return self.__listener()

    def close(self):
        if self._port:
            self._port.close()

    def __del__(self):
        self.close()


@Listeners.register("AudioChunk", "yarp")
class YarpAudioChunkListener(YarpImageListener):
    def __init__(self, name, in_port, carrier="", should_wait=False, channels=1, rate=44100, chunk=44100):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait,
                         width=chunk, height=channels, rgb=False)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._dummy_sound, self._dummy_port, self.__dummy_netconnect = [None] * 3
        ListenerWatchDog().add_listener(self)

    def establish(self):
        print("waiting for in_port: ", self.in_port + "_SND")
        while not yarp.Network.exists(self.in_port + "_SND"):
            yarp.Network.waitPort(self.in_port, quiet=True)
        print("connected to in_port: ", self.in_port + "_SND")

        if self.channels == -1 or self.rate == -1 or self.chunk == -1:
            # create a dummy sound object for transmitting the sound props. This could be cleaner but left for future impl.
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._dummy_port = yarp.Port()
            self._dummy_port.open(self.in_port + "_SND:in" + rnd_id)
            self.__dummy_netconnect = yarp.Network.connect(self.in_port + "_SND",
                                                             self.in_port + "_SND:in" + rnd_id,
                                                             self.carrier)
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
        return self._YarpImageListener__listener(), self.rate


@Listeners.register("NativeObject", "yarp")
class YarpNativeObjectListener(Listener):
    def __init__(self, name, in_port, carrier="", should_wait=False):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)

        self._json_object_hook = wrapify.utils.JsonDecodeHook().object_hook
        self._port, self.__netconnect = [None] * 2
        ListenerWatchDog().add_listener(self)

    def establish(self):
        print("waiting for in_port: ", self.in_port)
        while not yarp.Network.exists(self.in_port):
            yarp.Network.waitPort(self.in_port, quiet=True)
        print("connected to in_port: ", self.in_port)
        self._port = yarp.BufferedPortBottle()
        rnd_id = str(np.random.randint(100000, size=1)[0])
        self._port.open(self.in_port + ":in" + rnd_id)
        self.__netconnect = yarp.Network.connect(self.in_port, self.in_port + ":in" + rnd_id, self.carrier)

        self.established = True

    def listen(self):
        if not self.established:
            self.establish()
        obj = self._port.read(shouldWait=self.should_wait)
        if obj is not None:
            iobj = obj.get(0).asString()
            iobj = json.loads(iobj, object_hook=self._json_object_hook)
        else:
            iobj = None
        return iobj


@Listeners.register("Properties", "yarp")
class YarpPropertiesListener(Listener):
    def __init__(self, name, in_port, **kwargs):
        super().__init__(name, in_port, **kwargs)
        raise NotImplementedError
