import cv2

from wrapify.utils import JsonEncoder as json

try:
    import yarp
    yarp.Network.init()
except:
    print("Install YARP to use wrapify")


class Publishers(object):
    registry = {}

    @classmethod
    def register(cls, *args):
        def decorator(fn):
            cls.registry[args[0]] = fn
            return fn
        return decorator


# TODO (fabawi): Support multiple instance publishing of the same class,
#  currently only an issue with the output port naming convention
class Publisher(object):
    def __init__(self):
        pass

    def publish(self, obj):
        raise NotImplementedError


@Publishers.register("Image")
class YarpImagePublisher(Publisher):
    """
    The ImagePublisher using the BufferedPortImage construct assuming a cv2 image as an input
    """
    def __init__(self, name, out_port, carrier="", width=320, height=240, rgb=True):
        """
        Initializing the ImagePublisher
        :param name: Name of the publisher
        :param out_port: The published port name preceded by "/"
        :param carrier: For a list of carrier names, visit https://www.yarp.it/carrier_config.html. Default is "tcp"
        :param width: Image width
        :param height: Image height
        :param rgb: Transmits an RGB unsigned int image when "True". Transmits a float when "False"
        """
        super().__init__()
        self.__name__ = name
        self.width = width
        self.height = height
        if rgb:
            self._port = yarp.BufferedPortImageRgb()
            self._port.open(out_port)
            self.__netconnect__ = yarp.Network.connect(out_port, out_port + ":out", carrier)
        else:
            self._port = yarp.BufferedPortImageFloat()
            self._port.open(out_port)
            self.__netconnect__ = yarp.Network.connect(out_port, out_port + ":out", carrier)

    def publish(self, img):
        """
        Publish the image
        :param img: The cv2 image (img_width, img_height, channels)
        :return: None
        """
        img = cv2.resize(img, dsize=(self.width, self.height), interpolation=cv2.INTER_CUBIC)
        oimg = self._port.prepare()
        oimg.setExternal(img.data, img.shape[1], img.shape[0])
        self._port.write()

    def close(self):
        """
        Close the port
        :return: None
        """
        self._port.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@Publishers.register("AudioChunk")
class YarpAudioChunkPublisher(YarpImagePublisher):
    """
    Using the ImagePublisher to carry the sound signal. There are better alternatives (Sound) but
    don't seem to work with the python bindings at the moment
    """
    def __init__(self, name, out_port, carrier="", channels=1, rate=44100, chunk=-1):
        """
        Initializing the AudioPublisher
        :param name: Name of the publisher
        :param out_port: The published port name preceded by "/"
        :param carrier: For a list of carrier names, visit https://www.yarp.it/carrier_config.html. Default is "tcp"
        :param channels: Number of audio channels
        :param rate: Sampling rate of the audio signal
        :param chunk: Size of the chunk in samples. Transmits 1 second when chunk=rate
        """
        super().__init__(name, out_port, carrier=carrier, width=chunk, height=channels, rgb=False)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        # create a dummy sound object for transmitting the sound props. This could be cleaner but left for future impl.
        self._dummy_port = yarp.Port()
        self._dummy_port.open(out_port + "_SND")
        self.__dummy_netconnect__ = yarp.Network.connect(out_port + "_SND", out_port + "_SND:out", carrier)
        self._dummy_sound = yarp.Sound()
        self._dummy_sound.setFrequency(self.rate)
        self._dummy_sound.resize(self.chunk, self.channels)

    def publish(self, aud):
        """
        Publish the audio
        :param aud: The np audio signal ((audio_chunk, channels), samplerate)
        :return: None
        """
        aud, _ = aud
        if aud is not None:
            oaud = self._port.prepare()
            oaud.setExternal(aud.data, self.chunk if self.chunk != -1 else oaud.shape[1], self.channels)
            self._port.write()
            self._dummy_port.write(self._dummy_sound)



@Publishers.register("NativeObject")
class YarpNativeObjectPublisher(Publisher):
    """
        The NativeObjectPublisher using the BufferedPortBottle construct assuming a combination of python native objects
        and numpy arrays as input
        """
    def __init__(self, name, out_port, carrier=""):
        """
        Initializing the NativeObjectPublisher
        :param name: Name of the publisher
        :param out_port: The published port name preceded by "/"
        :param carrier: For a list of carrier names, visit https://www.yarp.it/carrier_config.html. Default is "tcp"
        """
        super().__init__()
        self.__name__ = name

        self._port = yarp.BufferedPortBottle()
        self._port.open(out_port)
        self.__netconnect__ = yarp.Network.connect(out_port, out_port + ":out", carrier)

    def publish(self, obj):
        obj = json.dumps(obj)
        oobj = self._port.prepare()
        oobj.clear()
        oobj.addString(obj)
        # print(oobj.get(0).asString())
        self._port.write()

@Publishers.register("Properties")
class YarpPropertiesPublisher(Publisher):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

