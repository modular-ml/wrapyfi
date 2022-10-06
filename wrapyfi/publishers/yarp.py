import logging
import json
import time

import numpy as np
import yarp

from wrapyfi.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapyfi.middlewares.yarp import YarpMiddleware
from wrapyfi.encoders import JsonEncoder


class YarpPublisher(Publisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        YarpMiddleware.activate()

    def await_connection(self, port, out_port=None, repeats=None):
        connected = False
        if out_port is None:
            out_port = self.out_port
        logging.info(f"Waiting for output connection: {out_port}")
        if repeats is None:
            if self.should_wait:
                repeats = -1
            else:
                repeats = 1
            while repeats > 0 or repeats <= -1:
                repeats -= 1
                connected = port.getOutputCount() < 1
                if connected:
                    break
                time.sleep(0.02)
        logging.info(f"Output connection established: {out_port}")
        return connected


@Publishers.register("NativeObject", "yarp")
class YarpNativeObjectPublisher(YarpPublisher):
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
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        self._port = self._netconnect = None
        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        self._port = yarp.BufferedPortBottle()
        self._port.open(self.out_port)
        self._netconnect = yarp.Network.connect(self.out_port, self.out_port_connect, self.carrier)
        established = self.await_connection(self._port, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, obj):
        if not self.established:
            established = self.establish()
            if not established:
                return
            else:
                time.sleep(0.2)
        obj = json.dumps(obj, cls=JsonEncoder)
        oobj = self._port.prepare()
        oobj.clear()
        oobj.addString(obj)
        self._port.write()


@Publishers.register("Image", "yarp")
class YarpImagePublisher(YarpPublisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, width=-1, height=-1, rgb=True, fp=False, **kwargs):
        """
        Initializing the ImagePublisher
        :param name: Name of the publisher
        :param out_port: The published port name preceded by "/"
        :param carrier: For a list of carrier names, visit https://www.yarp.it/carrier_config.html. Default is "tcp"
        :param out_port_connect: This is an optional port connection for listening devices (follows out_port format)
        :param width: Image width
        :param height: Image height
        :param rgb: Transmits an RGB image when "True", or mono image when "False"
        :param fp: Transmits 32-bit floating point image when "True", or 8-bit integer image when "False"
        """
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self._port = self._type = self._netconnect = None
        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        if self.rgb:
            self._port = yarp.BufferedPortImageRgbFloat() if self.fp else yarp.BufferedPortImageRgb()
        else:
            self._port = yarp.BufferedPortImageFloat() if self.fp else yarp.BufferedPortImageMono()
        self._type = np.float32 if self.fp else np.uint8
        self._port.open(self.out_port)
        self._netconnect = yarp.Network.connect(self.out_port, self.out_port_connect, self.carrier)
        established = self.await_connection(self._port, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, img):
        """
        Publish the image
        :param img: The cv2 image (img_height, img_width, channels)
        :return: None
        """
        if not self.established:
            established = self.establish()
            if not established:
                return
            else:
                time.sleep(0.2)
        if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
            raise ValueError("Incorrect image shape for publisher")
        img = np.require(img, dtype=self._type, requirements='C')
        yarp_img = self._port.prepare()
        yarp_img.resize(img.shape[1], img.shape[0])
        yarp_img.setExternal(img, img.shape[1], img.shape[0])
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
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, width=chunk, height=channels, rgb=False, fp=True)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self._dummy_sound = self._dummy_port = self._dummy_netconnect = None
        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        # create a dummy sound object for transmitting the sound props. This could be cleaner but left for future impl.
        self._dummy_port = yarp.Port()
        self._dummy_port.open(self.out_port + "_SND")
        self._dummy_netconnect = yarp.Network.connect(self.out_port + "_SND", self.out_port_connect + "_SND", self.carrier)
        self._dummy_sound = yarp.Sound()
        self._dummy_sound.setFrequency(self.rate)
        self._dummy_sound.resize(self.chunk, self.channels)
        established = self.await_connection(self._dummy_port, out_port=self.out_port + "_SND")
        if established:
            super(YarpAudioChunkPublisher, self).establish(repeats=repeats)
            self._dummy_port.write(self._dummy_sound)
        return self.check_establishment(established)

    def publish(self, aud):
        """
        Publish the audio
        :param aud: The np audio signal ((audio_chunk, channels), samplerate)
        :return: None
        """
        if not self.established:
            established = self.establish()
            if not established:
                return
            else:
                time.sleep(0.2)
        aud, _ = aud
        if aud is not None:
            oaud = self._port.prepare()
            oaud.setExternal(aud.data, self.chunk if self.chunk != -1 else oaud.shape[1], self.channels)
            self._port.write()

    def close(self):
        super().close()
        if self._dummy_port:
            self._dummy_port.close()


@Publishers.register("Properties", "yarp")
class YarpPropertiesPublisher(YarpPublisher):

    def __init__(self, name, out_port, **kwargs):
        super().__init__(name, out_port, **kwargs)
        raise NotImplementedError
