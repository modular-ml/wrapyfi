import logging
import json
from typing import Optional, Literal, Tuple

import numpy as np
import yarp

from wrapyfi.connect.servers import Server, Servers
from wrapyfi.middlewares.yarp import YarpMiddleware
from wrapyfi.utils.serialization_encoders import JsonEncoder, JsonDecodeHook


class YarpServer(Server):

    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: Literal["tcp", "udp", "mcast"] = "tcp",
        out_topic_connect: Optional[str] = None,
        persistent: bool = True,
        yarp_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the server.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param out_topic_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_topic. Default is None
        :param yarp_kwargs: dict: Additional kwargs for  the Yarp middleware
        :param kwargs: dict: Additional kwargs for the server
        """
        super().__init__(
            name,
            out_topic,
            carrier=carrier,
            out_topic_connect=out_topic_connect,
            **kwargs,
        )
        YarpMiddleware.activate(**yarp_kwargs or {})
        self.style = yarp.ContactStyle()
        self.style.persistent = persistent
        self.style.carrier = self.carrier

        self.persistent = persistent

    def close(self):
        """
        Close the server.
        """
        if hasattr(self, "_port") and self._port:
            if self._port is not None:
                self._port.close()

    def __del__(self):
        self.close()


@Servers.register("NativeObject", "yarp")
class YarpNativeObjectServer(YarpServer):

    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: Literal["tcp", "udp", "mcast"] = "tcp",
        out_topic_connect: Optional[str] = None,
        persistent: bool = True,
        serializer_kwargs: Optional[dict] = None,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Specific server handling native Python objects, serializing them to JSON strings for transmission.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param out_topic_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_topic. Default is None
        :param persistent: bool: Whether the server port should remain connected after closure. Default is True
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(
            name,
            out_topic,
            carrier=carrier,
            out_topic_connect=out_topic_connect,
            persistent=persistent,
            **kwargs,
        )
        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._port = self._netconnect = None

    def establish(self):
        """
        Establish the connection to the server.
        """
        self._port = yarp.RpcServer()
        self._port.open(self.out_topic)
        if self.style.persistent:
            self._netconnect = yarp.Network.connect(
                self.out_topic, self.out_topic_connect, self.style
            )
        else:
            self._netconnect = yarp.Network.connect(
                self.out_topic, self.out_topic_connect, self.carrier
            )

        self._netconnect = yarp.Network.connect(
            self.out_topic, self.out_topic_connect, self.carrier
        )
        if self.persistent:
            self.established = True

    def await_request(self, *args, **kwargs):
        """
        Await and deserialize the client's request, returning the extracted arguments and keyword arguments.
        The method blocks until a message is received, then attempts to deserialize it using the configured JSON decoder
        hook, returning the extracted arguments and keyword arguments.

        :return: Tuple[list, dict]: A tuple containing two items:
                 - A list of arguments extracted from the received message
                 - A dictionary of keyword arguments extracted from the received message
        """
        if not self.established:
            self.establish()
        try:
            obj_msg = yarp.Bottle()
            obj_msg.clear()
            request = False
            while not request:
                request = self._port.read(obj_msg, True)
            [args, kwargs] = json.loads(
                obj_msg.get(0).asString(),
                object_hook=self._plugin_decoder_hook,
                **self._deserializer_kwargs,
            )
            return args, kwargs
        except Exception as e:
            logging.error("[YARP] Service call failed: %s" % e)
            return [], {}

    def reply(self, obj):
        """
        Serialize the provided Python object to a JSON string and send it as a reply to the client.
        The method uses the configured JSON encoder for serialization before sending the resultant string to the client.

        :param obj: Any: The Python object to be serialized and sent
        """
        obj_str = json.dumps(
            obj,
            cls=self._plugin_encoder,
            **self._plugin_kwargs,
            serializer_kwrags=self._serializer_kwargs,
        )
        obj_msg = yarp.Bottle()
        obj_msg.clear()
        obj_msg.addString(obj_str)
        if self.persistent:
            self._port.reply(obj_msg)
        else:
            self._port.replyAndDrop(obj_msg)


@Servers.register("Image", "yarp")
class YarpImageServer(YarpNativeObjectServer):
    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: Literal["tcp", "udp", "mcast"] = "tcp",
        out_topic_connect: Optional[str] = None,
        persistent: bool = True,
        width: int = -1,
        height: int = -1,
        rgb: bool = True,
        fp: bool = False,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Specific server handling image data as numpy arrays, serializing them to JSON strings for transmission.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param out_topic_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_topic. Default is None
        :param persistent: bool: Whether the server port should remain connected after closure. Default is True
        :param width: int: Width of the image. Default is -1 (use the width of the received image)
        :param height: int: Height of the image. Default is -1 (use the height of the received image)
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        if "jpg" in kwargs:
            logging.warning(
                "[YARP] YARP currently does not support JPG encoding in REQ/REP. Using raw image."
            )
            kwargs.pop("jpg")
        super().__init__(
            name,
            out_topic,
            carrier=carrier,
            out_topic_connect=out_topic_connect,
            persistent=persistent,
            deserializer_kwargs=deserializer_kwargs,
            **kwargs,
        )
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp

    def reply(self, img: np.ndarray):
        """
        Serialize the provided image data and send it as a reply to the client.

        :param img: np.ndarray: Image to send formatted as a cv2 image - np.ndarray[img_height, img_width, channels]
        """
        if (
            0 < self.width != img.shape[1]
            or 0 < self.height != img.shape[0]
            or not (
                (img.ndim == 2 and not self.rgb)
                or (img.ndim == 3 and self.rgb and img.shape[2] == 3)
            )
        ):
            raise ValueError("Incorrect image shape for publisher")
        # img = np.require(img, dtype=self._type, requirements='C')
        super().reply(img)


@Servers.register("AudioChunk", "yarp")
class YarpAudioChunkServer(YarpNativeObjectServer):
    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: Literal["tcp", "udp", "mcast"] = "tcp",
        out_topic_connect: Optional[str] = None,
        persistent: bool = True,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = -1,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Specific server handling audio data as numpy arrays, serializing them to JSON strings for transmission.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param out_topic_connect: str: Name of the output topic connection alias '/' (e.g. '/topic:out') to connect to.
                                        None appends ':out' to the out_topic. Default is None
        :param persistent: bool: Whether the server port should remain connected after closure. Default is True
        :param channels: int: Number of channels in the audio. Default is 1
        :param rate: int: Sampling rate of the audio. Default is 44100
        :param chunk: int: Number of samples in the audio chunk. Default is -1 (use the chunk size of the received audio)
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(
            name,
            out_topic,
            carrier=carrier,
            out_topic_connect=out_topic_connect,
            persistent=persistent,
            deserializer_kwargs=deserializer_kwargs,
            **kwargs,
        )
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def reply(self, aud: Tuple[np.ndarray, int]):
        """
        Serialize the provided audio data and send it as a reply to the client.

        :param aud: Tuple[np.ndarray, int]: Audio chunk to publish formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        aud, rate = aud
        if aud is None:
            return
        if 0 < self.rate != rate:
            raise ValueError("Incorrect audio rate for server reply")
        chunk, channels = aud.shape if len(aud.shape) > 1 else (aud.shape[0], 1)
        self.chunk = chunk if self.chunk == -1 else self.chunk
        self.channels = channels if self.channels == -1 else self.channels
        if 0 < self.chunk != chunk or 0 < self.channels != channels:
            raise ValueError("Incorrect audio shape for publisher")
        aud = np.require(aud, dtype=np.float32, requirements="C")
        super().reply((chunk, channels, rate, aud))
