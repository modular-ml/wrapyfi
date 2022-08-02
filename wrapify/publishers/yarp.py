import json
import logging
import cv2

try:
    import yarp
    yarp.Network.init()
    logging.info("YARP is found and can be used")
except:
    # print("Install YARP to use wrapify")
    logging.warning("YARP is not found and cannot be used")
    pass

from wrapify.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapify.utils import JsonEncoder


@Publishers.register("Image", "yarp")
class YarpImagePublisher(Publisher):
    """
    The ImagePublisher using the BufferedPortImage construct assuming a cv2 image as an input
    """
    def __init__(self, name, out_port, carrier="", out_port_connect=None, width=320, height=240, rgb=True, **kwargs):
        """
        Initializing the ImagePublisher
        :param name: Name of the publisher
        :param out_port: The published port name preceded by "/"
        :param carrier: For a list of carrier names, visit https://www.yarp.it/carrier_config.html. Default is "tcp"
        :param out_port_connect: This is an optional port connection for listening devices (follows out_port format)
        :param width: Image width
        :param height: Image height
        :param rgb: Transmits an RGB unsigned int image when "True". Transmits a float when "False"
        """
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect)
        self.width = width
        self.height = height
        self.rgb = rgb

        self._port, self.__netconnect = [None] * 2
        PublisherWatchDog().add_publisher(self)

    def establish(self):
        if self.rgb:
            self._port = yarp.BufferedPortImageRgb()
            self._port.open(self.out_port)
            self.__netconnect = yarp.Network.connect(self.out_port, self.out_port_connect, self.carrier)
        else:
            self._port = yarp.BufferedPortImageFloat()
            self._port.open(self.out_port)
            self.__netconnect = yarp.Network.connect(self.out_port, self.out_port_connect, self.carrier)

        self.established = True

    def publish(self, img):
        """
        Publish the image
        :param img: The cv2 image (img_width, img_height, channels)
        :return: None
        """
        if not self.established:
            self.establish()
        img = cv2.resize(img, dsize=(self.width, self.height), interpolation=cv2.INTER_CUBIC)
        oimg = self._port.prepare()
        oimg.setExternal(img.data, img.shape[1], img.shape[0])
        self._port.write()

    def close(self):
        """
        Close the port
        :return: None
        """
        if self._port:
            self._port.close()

    def __del__(self):
        self.close()


@Publishers.register("AudioChunk", "yarp")
class YarpAudioChunkPublisher(YarpImagePublisher):
    """
    Using the ImagePublisher to carry the sound signal. There are better alternatives (Sound) but
    don't seem to work with the python bindings at the moment
    """
    def __init__(self, name, out_port, carrier="", out_port_connect=None, channels=1, rate=44100, chunk=-1, **kwargs):
        """
        Initializing the AudioPublisher
        :param name: Name of the publisher
        :param out_port: The published port name preceded by "/"
        :param carrier: For a list of carrier names, visit https://www.yarp.it/carrier_config.html. Default is "tcp"
        :param out_port_connect: This is an optional port connection for listening devices (follows out_port format)
        :param channels: Number of audio channels
        :param rate: Sampling rate of the audio signal
        :param chunk: Size of the chunk in samples. Transmits 1 second when chunk=rate
        """
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect,
                         width=chunk, height=channels, rgb=False)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._dummy_sound, self._dummy_port, self.__dummy_netconnect = [None] * 3
        PublisherWatchDog().add_publisher(self)

    def establish(self):
        # create a dummy sound object for transmitting the sound props. This could be cleaner but left for future impl.
        self._dummy_port = yarp.Port()
        self._dummy_port.open(self.out_port + "_SND")
        self.__dummy_netconnect = yarp.Network.connect(self.out_port + "_SND", self.out_port_connect + "_SND",
                                                         self.carrier)
        self._dummy_sound = yarp.Sound()
        self._dummy_sound.setFrequency(self.rate)
        self._dummy_sound.resize(self.chunk, self.channels)

        super(YarpAudioChunkPublisher, self).establish()

    def publish(self, aud):
        """
        Publish the audio
        :param aud: The np audio signal ((audio_chunk, channels), samplerate)
        :return: None
        """
        if not self.established:
            self.establish()
        aud, _ = aud
        if aud is not None:
            oaud = self._port.prepare()
            oaud.setExternal(aud.data, self.chunk if self.chunk != -1 else oaud.shape[1], self.channels)
            self._port.write()
            self._dummy_port.write(self._dummy_sound)


@Publishers.register("NativeObject", "yarp")
class YarpNativeObjectPublisher(Publisher):
    """
        The NativeObjectPublisher using the BufferedPortBottle construct assuming a combination of python native objects
        and numpy arrays as input
        """
    def __init__(self, name, out_port, carrier="", out_port_connect=None, **kwargs):
        """
        Initializing the NativeObjectPublisher
        :param name: Name of the publisher
        :param out_port: The published port name preceded by "/"
        :param carrier: For a list of carrier names, visit https://www.yarp.it/carrier_config.html. Default is "tcp"
        :param out_port_connect: This is an optional port connection for listening devices (follows out_port format)
        """
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect)

        self._port, self.__netconnect = [None] * 2
        PublisherWatchDog().add_publisher(self)

    def establish(self):
        self._port = yarp.BufferedPortBottle()
        self._port.open(self.out_port)
        self.__netconnect = yarp.Network.connect(self.out_port, self.out_port_connect, self.carrier)

        self.established = True

    def publish(self, obj):
        if not self.established:
            self.establish()
        obj = json.dumps(obj, cls=JsonEncoder)
        oobj = self._port.prepare()
        oobj.clear()
        oobj.addString(obj)
        # print(oobj.get(0).asString())
        self._port.write()


@Publishers.register("Properties", "yarp")
class YarpPropertiesPublisher(Publisher):
    def __init__(self, name, out_port, **kwargs):
        super().__init__(name, out_port, **kwargs)
        raise NotImplementedError
