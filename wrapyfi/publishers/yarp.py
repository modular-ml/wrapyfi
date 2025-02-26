import logging
import json
import time
import base64
import io
from typing import Optional, Literal, Tuple, Union

import numpy as np
import yarp

from wrapyfi.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapyfi.middlewares.yarp import YarpMiddleware
from wrapyfi.utils.serialization_encoders import JsonEncoder
from wrapyfi.utils.image_encoders import JpegEncoder


WATCHDOG_POLL_REPEAT = None


class YarpPublisher(Publisher):

    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: Literal["tcp", "udp", "mcast"] = "tcp",
        should_wait: bool = True,
        persistent: bool = True,
        out_topic_connect: Optional[str] = None,
        yarp_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the publisher.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param persistent: bool: Whether the publisher port should remain connected after closure. Default is True
        :param out_topic_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_topic. Default is None
        :param yarp_kwargs: dict: Additional kwargs for  the Yarp middleware
        :param kwargs: dict: Additional kwargs for the publisher
        """
        super().__init__(
            name, out_topic, carrier=carrier, should_wait=should_wait, **kwargs
        )
        YarpMiddleware.activate(**yarp_kwargs or {})

        self.style = yarp.ContactStyle()
        self.style.persistent = persistent
        self.style.carrier = self.carrier

        self.out_topic_connect = (
            out_topic + ":out" if out_topic_connect is None else out_topic_connect
        )

    def await_connection(
        self, port, out_topic: Optional[str] = None, repeats: Optional[int] = None
    ):
        """
        Wait for at least one subscriber to connect to the publisher.

        :param port: yarp.Port: Port to await connection to
        :param out_topic: str: Name of the output topic
        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        connected = False
        if out_topic is None:
            out_topic = self.out_topic
        logging.info(f"[YARP] Waiting for output connection: {out_topic}")
        if repeats is None:
            repeats = -1 if self.should_wait else 1
            while repeats > 0 or repeats <= -1:
                repeats -= 1
                # allowing should_wait into the loop for consistency with other mware publishers only
                connected = port.getOutputCount() > 0 or not self.should_wait
                if connected:
                    logging.info(f"[YARP] Output connection established: {out_topic}")
                    break
                time.sleep(0.02)
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

    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: Literal["tcp", "udp", "mcast"] = "tcp",
        should_wait: bool = True,
        persistent: bool = True,
        out_topic_connect: str = None,
        serializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The NativeObject publisher using the BufferedPortBottle string construct assuming a combination of python native objects
        and numpy arrays as input. Serializes the data (including plugins) using the encoder and sends it as a string.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param persistent: bool: Whether the publisher port should remain connected after closure. Default is True
        :param out_topic_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_topic. Default is None
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        """
        super().__init__(
            name,
            out_topic,
            carrier=carrier,
            should_wait=should_wait,
            persistent=persistent,
            out_topic_connect=out_topic_connect,
            **kwargs,
        )
        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}

        self._port = self._netconnect = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        self._port = yarp.BufferedPortBottle()
        self._port.open(self.out_topic)
        if self.style.persistent:
            self._netconnect = yarp.Network.connect(
                self.out_topic, self.out_topic_connect, self.style
            )
        else:
            self._netconnect = yarp.Network.connect(
                self.out_topic, self.out_topic_connect, self.carrier
            )
        established = self.await_connection(self._port, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, obj):
        """
        Publish the object to the middleware.

        :param obj: object: Object to publish
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)
        obj_str = json.dumps(
            obj,
            cls=self._plugin_encoder,
            **self._plugin_kwargs,
            serializer_kwrags=self._serializer_kwargs,
        )
        obj_port = self._port.prepare()
        obj_port.clear()
        obj_port.addString(obj_str)
        self._port.write()


@Publishers.register("Image", "yarp")
class YarpImagePublisher(YarpPublisher):

    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: Literal["tcp", "udp", "mcast"] = "tcp",
        should_wait: bool = True,
        persistent: bool = True,
        out_topic_connect: Optional[str] = None,
        width: int = -1,
        height: int = -1,
        rgb: bool = True,
        fp: bool = False,
        jpg: Union[bool, dict] = False,
        **kwargs,
    ):
        """
        The Image publisher using the BufferedPortImage construct assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param persistent: bool: Whether the publisher port should remain connected after closure. Default is True
        :param out_topic_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_topic. Default is None
        :param width: int: Width of the image. Default is -1 meaning the width of the input image
        :param height: int: Height of the image. Default is -1 meaning the height of the input image
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: Union[bool, dict]: If True, compress as JPG with default settings. If a dict, pass arguments to JpegEncoder. Default is False
        """
        super().__init__(
            name,
            out_topic,
            carrier=carrier,
            should_wait=should_wait,
            persistent=persistent,
            out_topic_connect=out_topic_connect,
            **kwargs,
        )
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        if self.jpg:
            self._image_encoder = JpegEncoder(
                **(self.jpg if isinstance(self.jpg, dict) else {})
            )

        self._port = self._type = self._netconnect = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        if self.jpg:
            self._port = yarp.BufferedPortBottle()
        elif self.rgb:
            self._port = (
                yarp.BufferedPortImageRgbFloat()
                if self.fp
                else yarp.BufferedPortImageRgb()
            )
        else:
            self._port = (
                yarp.BufferedPortImageFloat()
                if self.fp
                else yarp.BufferedPortImageMono()
            )
        self._type = np.float32 if self.fp else np.uint8
        self._port.open(self.out_topic)
        if self.style.persistent:
            self._netconnect = yarp.Network.connect(
                self.out_topic, self.out_topic_connect, self.style
            )
        else:
            self._netconnect = yarp.Network.connect(
                self.out_topic, self.out_topic_connect, self.carrier
            )
        established = self.await_connection(self._port, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, img: np.ndarray):
        """
        Publish the image to the middleware.

        :param img: np.ndarray: Image to publish formatted as a cv2 image np.ndarray[img_height, img_width, channels] to publish
        """
        if img is None:
            return

        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)

        if (
            0 < self.width != img.shape[1]
            or 0 < self.height != img.shape[0]
            or not (
                (img.ndim == 2 and not self.rgb)
                or (img.ndim == 3 and self.rgb and img.shape[2] == 3)
            )
        ):
            raise ValueError("Incorrect image shape for publisher")
        img = np.require(img, dtype=self._type, requirements="C")

        if self.jpg:
            img_str = self._image_encoder.encode_jpg_image(
                img, return_numpy=True
            ).tostring()
            with io.BytesIO() as memfile:
                np.save(memfile, img_str)
                img_str = base64.b64encode(memfile.getvalue()).decode("ascii")
            img_port = self._port.prepare()
            img_port.clear()
            img_port.addString(img_str)
            self._port.write()

        else:
            img_port = self._port.prepare()
            img_port.resize(img.shape[1], img.shape[0])
            img_port.setExternal(img.data, img.shape[1], img.shape[0])
            self._port.write()


@Publishers.register("AudioChunk", "yarp")
class YarpAudioChunkPublisher(YarpPublisher):

    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: Literal["tcp", "udp", "mcast"] = "tcp",
        should_wait: bool = True,
        persistent: bool = True,
        out_topic_connect: Optional[str] = None,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = -1,
        **kwargs,
    ):
        """
         The AudioChunk publisher using the Sound construct assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param persistent: bool: Whether the publisher port should remain connected after closure. Default is True
        :param out_topic_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_topic. Default is None
        :param channels: int: Number of channels. Default is 1
        :param rate: int: Sampling rate. Default is 44100
        :param chunk: int: Chunk size. Default is -1 meaning that the chunk size is not fixed
        """
        super().__init__(
            name,
            out_topic,
            carrier=carrier,
            should_wait=should_wait,
            out_topic_connect=out_topic_connect,
            persistent=persistent,
            **kwargs,
        )
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._sound_msg = self._port = self._netconnect = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        # create a dummy sound object for transmitting the sound props. This could be cleaner but left for future impl.
        self._port = yarp.Port()
        self._port.open(self.out_topic)
        self._netconnect = yarp.Network.connect(
            self.out_topic, self.out_topic_connect, self.carrier
        )
        self._sound_msg = yarp.Sound()
        self._sound_msg.setFrequency(self.rate)
        self._sound_msg.resize(self.chunk, self.channels)
        established = self.await_connection(self._port, out_topic=self.out_topic)
        if established:
            self._port.write(self._sound_msg)

        return self.check_establishment(established)

    def publish(self, aud: Tuple[np.ndarray, int]):
        """
        Publish the audio chunk to the middleware.

        :param aud: Tuple[np.ndarray, int]: Audio chunk to publish formatted as ((chunk_size, channels), samplerate)
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)

        aud, rate = aud
        if aud is None:
            return
        if 0 < self.rate != rate:
            raise ValueError("Incorrect audio rate for publisher")
        chunk, channels = aud.shape if len(aud.shape) > 1 else (aud.shape[0], 1)
        self.chunk = chunk if self.chunk == -1 else self.chunk
        self.channels = channels if self.channels == -1 else self.channels
        if 0 < self.chunk != chunk or 0 < self.channels != channels:
            raise ValueError("Incorrect audio shape for publisher")
        aud = np.require(aud, dtype=np.float32, requirements="C")

        for i in range(aud.size):
            self._sound_msg.set(
                int(aud.data[i] * 32767), i
            )  # Convert float samples to 16-bit int

        self._port.write(self._sound_msg)


@Publishers.register("Properties", "yarp")
class YarpPropertiesPublisher(YarpPublisher):

    def __init__(self, name, out_topic, **kwargs):
        super().__init__(name, out_topic, **kwargs)
        raise NotImplementedError
