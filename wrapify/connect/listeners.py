import time

import cv2
import numpy as np

from wrapify.utils import JsonEncoder as json
from wrapify.utils import SingletonOptimized

try:
    import yarp
    yarp.Network.init()
except:
    print("Install YARP to use wrapify")

class ListenerWatchDog(metaclass=SingletonOptimized):
    def __init__(self):
        self.listener_ring = []

    def add_listener(self, listener, listener_kwargs):
        self.listener_ring.append((listener, listener_kwargs))

    def scan(self):
        while self.listener_ring:
            for listener in self.listener_ring:
                found_listener = listener[0].establish(**listener[1])
                if found_listener:
                    self.listener_ring.remove(listener)


class Listeners(object):
    registry = {}

    @classmethod
    def register(cls, *args):
        def decorator(fn):
            cls.registry[args[0]] = fn
            return fn
        return decorator


class Listener(object):
    def __init__(self):
        pass

    def establish(self):
        raise NotImplementedError

    def listen(self):
        raise NotImplementedError

@Listeners.register("Image")
class YarpImageListener(Listener):
    def __init__(self, name, in_port, carrier="", width=320, height=240, rgb=True):
        super().__init__()
        self.__name__ = name
        print("waiting for in_port: ", in_port)
        while not yarp.Network.exists(in_port):
            yarp.Network.waitPort(in_port, quiet=True)
        print("connected to in_port: ", in_port)
        if rgb:
            self._port = yarp.BufferedPortImageRgb()
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._port.open(in_port + ":in" + rnd_id)
            self.__netconnect__ = yarp.Network.connect(in_port, in_port + ":in" + rnd_id, carrier)
            if height != -1 and width != -1:
                self._iarray = np.zeros((height, width, 3), dtype=np.uint8)
            self.__type__ = np.uint8
        else:
            self._port = yarp.BufferedPortImageFloat()
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._port.open(in_port + ":in" + rnd_id)
            self.__netconnect__ = yarp.Network.connect(in_port, in_port + ":in" + rnd_id, carrier)
            if height != -1 and width != -1:
                self._iarray = np.zeros((height, width), dtype=np.float32)
            self.__type__ = np.float32

        # set the listener
        if width == -1 and height == -1:
            self.__listener__ = self.__listen_unk_width_height__
        elif width == -1:
            self.__listener__ = self.__listen_unk_width__
        elif height == -1:
            self.__listener__ = self.__listen_unk_height__
        else:
            self.__listener__ = self.__listen__

    def __listen_unk_width__(self):
        # TODO (fabawi): dynamic width only
        img = self._port.read(shouldWait=False)
        if img is not None:
            self._iarray = np.zeros((img.height(), img.width()), dtype=self.__type__)
            img.setExternal(self._iarray.data,
                            img.width(), img.height())
        return self._iarray

    def __listen_unk_height__(self):
        # TODO (fabawi): dynamic height only
        img = self._port.read(shouldWait=False)
        if img is not None:
            self._iarray = np.zeros((img.height(), img.width()), dtype=self.__type__)
            img.setExternal(self._iarray.data,
                            img.width(), img.height())
        return self._iarray

    def __listen_unk_width_height__(self):
        img = self._port.read(shouldWait=False)
        if img is not None:
            self._iarray = np.zeros((img.height(), img.width()), dtype=self.__type__)
            img.setExternal(self._iarray.data,
                            img.width(), img.height())
        return self._iarray

    def __listen__(self):
        img = self._port.read(shouldWait=False)
        if img is not None:
            img.setExternal(self._iarray.data,
                        self._iarray.shape[1], self._iarray.shape[0])
        return self._iarray

    def listen(self):
        return self.__listener__()

    def close(self):
        self._port.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@Listeners.register("AudioChunk")
class YarpAudioChunkListener(YarpImageListener):
    def __init__(self, name, in_port, carrier="", channels=1, rate=44100, chunk=44100):
        self.rate = rate
        print("waiting for in_port: ", in_port + "_SND")
        while not yarp.Network.exists(in_port + "_SND"):
            yarp.Network.waitPort(in_port, quiet=True)
        print("connected to in_port: ", in_port + "_SND")

        if channels == -1 or rate == -1 or chunk == -1:
            # create a dummy sound object for transmitting the sound props. This could be cleaner but left for future impl.
            rnd_id = str(np.random.randint(100000, size=1)[0])
            self._dummy_port = yarp.Port()
            self._dummy_port.open(in_port + "_SND:in" + rnd_id)
            self.__dummy_netconnect__ = yarp.Network.connect(in_port + "_SND", in_port + "_SND:in" + rnd_id, carrier)
            self._dummy_sound = yarp.Sound()
            self.rate = self._dummy_sound.getFrequency()
            self.channels = channels = self._dummy_sound.getChannels()
            self.chunk = chunk = self._dummy_sound.getSamples()

        super().__init__(name, in_port, carrier=carrier, width=chunk, height=channels, rgb=False)

    def listen(self):
        return self.__listener__(), self.rate


@Listeners.register("NativeObject")
class YarpNativeObjectListener(Listener):
    def __init__(self, name, in_port, carrier=""):
        super().__init__()
        self.__name__ = name
        print("waiting for in_port: ", in_port)
        while not yarp.Network.exists(in_port):
            yarp.Network.waitPort(in_port, quiet=True)
        print("connected to in_port: ", in_port)
        self._port = yarp.BufferedPortBottle()
        rnd_id = str(np.random.randint(100000, size=1)[0])
        self._port.open(in_port + ":in" + rnd_id)
        self.__netconnect__ = yarp.Network.connect(in_port, in_port + ":in" + rnd_id, carrier)

    def listen(self):
        obj = self._port.read(shouldWait=True)
        if obj is not None:
            iobj = obj.get(0).asString()
            iobj = json.loads(iobj)
        else:
            iobj = None
        return iobj

@Listeners.register("Properties")
class YarpPropertiesListener(Listener):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
