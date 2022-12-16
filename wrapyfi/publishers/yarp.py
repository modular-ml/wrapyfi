import logging
import json
import time
import base64
import io
from typing import Optional, Literal, Tuple

import numpy as np
import cv2
import yarp

from wrapyfi.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapyfi.middlewares.yarp import YarpMiddleware
from wrapyfi.encoders import JsonEncoder


WATCHDOG_POLL_REPEAT = None


class YarpPublisher(Publisher):

    def __init__(self, name: str, out_port: str, carrier: Literal["tcp", "udp", "mcast"] = "tcp", should_wait: bool = True,
                 out_port_connect: Optional[str] = None, yarp_kwargs: Optional[dict] = None, **kwargs):
        """
        Initialize the publisher

        :param name: str: Name of the publisher
        :param out_port: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param out_port_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_port. Default is None
        :param yarp_kwargs: dict: Additional kwargs for  the Yarp middleware
        :param kwargs: dict: Additional kwargs for the publisher
        """
        super().__init__(name, out_port, carrier=carrier, should_wait=should_wait, **kwargs)
        YarpMiddleware.activate(**yarp_kwargs or {})

        self.out_port_connect = out_port + ":out" if out_port_connect is None else out_port_connect

    def await_connection(self, port, out_port: Optional[str] = None, repeats: Optional[int] = None):
        """
        Wait for at least one subscriber to connect to the publisher

        :param port: yarp.Port: Port to await connection to
        :param out_port: str: Name of the output topic
        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
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

    def close(self):
        """
        Close the publisher
        """
        if hasattr(self, "_port") and self._port:
            if self._port is not None:
                self._port.close()

    def __del__(self):
        self.close()


@Publishers.register("NativeObject", "yarp")
class YarpNativeObjectPublisher(YarpPublisher):

    def __init__(self, name: str, out_port: str, carrier: Literal["tcp", "udp", "mcast"] = "tcp", should_wait: bool = True,
                 out_port_connect: str = None, serializer_kwargs: Optional[dict] = None, **kwargs):
        """
        The NativeObject publisher using the BufferedPortBottle string construct assuming a combination of python native objects
        and numpy arrays as input. Serializes the data (including plugins) using the encoder and sends it as a string

        :param name: str: Name of the publisher
        :param out_port: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param out_port_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_port. Default is None
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        """
        super().__init__(name, out_port, carrier=carrier, should_wait=should_wait, out_port_connect=out_port_connect, **kwargs)
        self._port = self._netconnect = None

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        self._port = yarp.BufferedPortBottle()
        self._port.open(self.out_port)
        self._netconnect = yarp.Network.connect(self.out_port, self.out_port_connect, self.carrier)
        established = self.await_connection(self._port, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, obj):
        """
        Publish the object to the middleware

        :param obj: object: Object to publish
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)
        obj_str = json.dumps(obj, cls=self._plugin_encoder, **self._plugin_kwargs,
                             serializer_kwrags=self._serializer_kwargs)
        oobj = self._port.prepare()
        oobj.clear()
        oobj.addString(obj_str)
        self._port.write()


@Publishers.register("Image", "yarp")
class YarpImagePublisher(YarpPublisher):

    def __init__(self, name: str, out_port: str, carrier: Literal["tcp", "udp", "mcast"] = "tcp", should_wait: bool = True,
                 out_port_connect: Optional[str] = None, width: int = -1, height: int = -1,
                 rgb: bool = True, fp: bool = False, jpg: bool = False, **kwargs):
        """
        The Image publisher using the BufferedPortImage construct assuming a numpy array as input

        :param name: str: Name of the publisher
        :param out_port: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param out_port_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_port. Default is None
        :param width: int: Width of the image. Default is -1 meaning the width of the input image
        :param height: int: Height of the image. Default is -1 meaning the height of the input image
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: bool: True if the image should be compressed as JPG. Default is False
        """
        super().__init__(name, out_port, carrier=carrier, should_wait=should_wait, out_port_connect=out_port_connect, **kwargs)

        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        self._port = self._type = self._netconnect = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        if self.jpg:
            self._port = yarp.BufferedPortBottle()
        elif self.rgb:
            self._port = yarp.BufferedPortImageRgbFloat() if self.fp else yarp.BufferedPortImageRgb()
        else:
            self._port = yarp.BufferedPortImageFloat() if self.fp else yarp.BufferedPortImageMono()
        self._type = np.float32 if self.fp else np.uint8
        self._port.open(self.out_port)
        self._netconnect = yarp.Network.connect(self.out_port, self.out_port_connect, self.carrier)
        established = self.await_connection(self._port, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, img: np.ndarray):
        """
        Publish the image to the middleware
        :param img: np.ndarray: Image to publish formatted as a cv2 image np.ndarray[img_height, img_width, channels] to publish
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)

        if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
            raise ValueError("Incorrect image shape for publisher")
        img = np.require(img, dtype=self._type, requirements='C')

        if self.jpg:
            img_compressed = cv2.imencode('.jpg', img)[1]
            with io.BytesIO() as memfile:
                np.save(memfile, img_compressed)
                img_str = base64.b64encode(memfile.getvalue()).decode('ascii')
            oobj = self._port.prepare()
            oobj.clear()
            oobj.addString(img_str)
            self._port.write()

        else:
            img_msg = self._port.prepare()
            img_msg.resize(img.shape[1], img.shape[0])
            img_msg.setExternal(img.data, img.shape[1], img.shape[0])
            self._port.write()


@Publishers.register("AudioChunk", "yarp")
class YarpAudioChunkPublisher(YarpImagePublisher):

    def __init__(self, name: str, out_port: str, carrier: Literal["tcp", "udp", "mcast"] = "tcp", should_wait: bool = True,

                 out_port_connect: Optional[str] = None,
                 channels: int = 1, rate: int = 44100, chunk: int = -1, **kwargs):
        """
         The AudioChunk publisher using the BufferedPortImage construct assuming a numpy array as input

        :param name: str: Name of the publisher
        :param out_port: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param out_port_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_port. Default is None
        :param channels: int: Number of channels. Default is 1
        :param rate: int: Sampling rate. Default is 44100
        :param chunk: int: Chunk size. Default is -1 meaning that the chunk size is not fixed
        """
        super().__init__(name, out_port, carrier=carrier, should_wait=should_wait, out_port_connect=out_port_connect,
                         width=chunk, height=channels, rgb=False, fp=True, jpg=False, **kwargs)

        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self._dummy_sound = self._dummy_port = self._dummy_netconnect = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
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

    def publish(self, aud: Tuple[np.ndarray, int]):
        """
        Publish the audio chunk to the middleware

        :param aud: (np.ndarray, int): Audio chunk to publish formatted as ((chunk_size, channels), samplerate)
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
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
        """
        Close the publisher connection to the yarp Sound port. This is not used at the moment, but left for future impl.
        """
        super().close()
        if self._dummy_port is not None:
            self._dummy_port.close()


@Publishers.register("Properties", "yarp")
class YarpPropertiesPublisher(YarpPublisher):

    def __init__(self, name, out_port, **kwargs):
        super().__init__(name, out_port, **kwargs)
        raise NotImplementedError
