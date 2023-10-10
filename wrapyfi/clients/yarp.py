import logging
import json
import time
from typing import Optional, Literal
import queue

import numpy as np
import yarp

from wrapyfi.connect.clients import Client, Clients
from wrapyfi.middlewares.yarp import YarpMiddleware
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class YarpClient(Client):

    def __init__(self, name: str, in_topic: str, carrier: Literal["tcp", "udp", "mcast"] = "tcp",
                 persistent: bool = True, yarp_kwargs: Optional[dict] = None, **kwargs):
        """
        Initialize the client.

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param persistent: bool: Whether to keep the service connection alive across multiple service calls. Default is True
        :param yarp_kwargs: dict: Additional kwargs for  the Yarp middleware
        :param kwargs: dict: Additional kwargs for the client
        """
        super().__init__(name, in_topic, carrier=carrier, **kwargs)
        YarpMiddleware.activate(**yarp_kwargs or {})

        self.persistent = persistent
    def close(self):
        """
        Close the client.
        """
        if hasattr(self, "_port") and self._port:
            if self._port is not None:
                self._port.close()

    def __del__(self):
        self.close()


@Clients.register("NativeObject", "yarp")
class YarpNativeObjectClient(YarpClient):
    def __init__(self, name: str, in_topic: str, carrier: Literal["tcp", "udp", "mcast"] = "tcp",
                 persistent: bool = True,
                 serializer_kwargs: Optional[dict] = None, deserializer_kwargs: Optional[dict] = None, **kwargs):
        """
        The NativeObject listener using the YARP Bottle construct assuming the data is serialized as a JSON string.
        Deserializes the data (including plugins) using the decoder and parses it to a Python object.

        :param name: str: Name of the client
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param serializer_kwargs: dict: Additional kwargs for the serializer. Defaults to None
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        :param persistent: bool: Whether to keep the service connection alive across multiple service calls. Default is True
        """
        super().__init__(name, in_topic, carrier=carrier, persistent=persistent, **kwargs)
        self._port = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

    def establish(self):
        """
        Establish the client's connection to the YARP service.
        """
        while not yarp.Network.exists(self.in_topic):
            logging.info(f"[YARP] Waiting for input port: {self.in_topic}")
            time.sleep(0.2)
        self._port = yarp.RpcClient()
        rnd_id = str(np.random.randint(100000, size=1)[0])
        self._port.open(self.in_topic + ":in" + rnd_id)
        self._port.addOutput(self.in_topic, self.carrier)
        if self.persistent:
            self.established = True

    def request(self, *args, **kwargs):
        """
        Send a request to the YARP service.

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        :return: Any: The response from the YARP service
        """
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except Exception as e:
            logging.error("[YARP] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        """
        Internal method to send a request to the YARP service.

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        """
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = yarp.Bottle()
        args_msg.clear()
        args_msg.addString(args_str)

        msg = yarp.Bottle()
        msg.clear()

        self._port.write(args_msg, msg)
        obj = json.loads(msg.get(0).asString(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        self._queue.put(obj, block=False)

    def _await_reply(self):
        """
        Wait for and return the reply from the YARP service.

        :return: Any: The response from the YARP service
        """
        try:
            reply = self._queue.get(block=True)
            return reply
        except queue.Full:
            logging.warning(f"[YARP] Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
            return None


@Clients.register("Image", "yarp")
class YarpImageClient(YarpNativeObjectClient):
    def __init__(self, name: str, in_topic: str, carrier: Literal["tcp", "udp", "mcast"] = "tcp",
                 width: int = -1, height: int = -1, rgb: bool = True, fp: bool = False,
                 persistent: bool = True, serializer_kwargs: Optional[dict] = None, **kwargs):
        """
        The Image client using the YARP Bottle construct parsed to a numpy array.

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param width: int: The width of the image. Default is -1
        :param height: int: The height of the image. Default is -1
        :param rgb: bool: Whether the image is RGB. Default is True
        :param fp: bool: Whether to utilize floating-point precision. Default is False
        :param persistent: bool: Whether to keep the service connection alive across multiple service calls. Default is True
        :param serializer_kwargs: dict: Additional kwargs for the serializer. Defaults to None
        :param kwargs: dict: Additional kwargs
        """
        super().__init__(name, in_topic, carrier=carrier, persistent=persistent, serializer_kwargs=serializer_kwargs, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp

    def _request(self, *args, **kwargs):
        """
        Internal method to send a request to the YARP service.

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        """
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = yarp.Bottle()
        args_msg.clear()
        args_msg.addString(args_str)
        msg = yarp.Bottle()
        msg.clear()
        self._port.write(args_msg, msg)
        img = json.loads(msg.get(0).asString(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        height, width, channels = img.shape
        if 0 < self.width != width or 0 < self.height != height:
            raise ValueError("Incorrect image shape for client")
        else:
            self._queue.put(img, block=False)


@Clients.register("AudioChunk", "yarp")
class YarpAudioChunkClient(YarpNativeObjectClient):
    def __init__(self, name: str, in_topic: str, carrier: Literal["tcp", "udp", "mcast"] = "tcp",
                 channels: int = 1, rate: int = 44100, chunk: int = -1,
                 persistent: bool = True, serializer_kwargs: Optional[dict] = None, **kwargs):
        """
        The Image client using the YARP Bottle construct parsed to a numpy array.

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param channels: int: Number of audio channels. Default is 1
        :param rate: int: Sampling rate of the audio. Default is 44100
        :param chunk: int: The size of audio chunks. Default is -1
        :param persistent: bool: Whether to keep the service connection alive across multiple service calls. Default is True
        :param serializer_kwargs: dict: Additional kwargs for the serializer. Defaults to None
        :param kwargs: dict: Additional kwargs
        """
        super().__init__(name, in_topic, carrier=carrier, persistent=persistent, serializer_kwargs=serializer_kwargs, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def _request(self, *args, **kwargs):
        """
        Internal method to send a request to the YARP service.

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        """
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = yarp.Bottle()
        args_msg.clear()
        args_msg.addString(args_str)
        msg = yarp.Bottle()
        msg.clear()
        self._port.write(args_msg, msg)
        chunk, channels, rate, aud = json.loads(msg.get(0).asString(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        if 0 < self.rate != rate:
            raise ValueError("Incorrect audio rate for client")
        if 0 < self.chunk != chunk or self.channels != channels or aud.size != chunk * channels:
            raise ValueError("Incorrect audio shape for client")
        else:
            if aud.shape[1] == 1:
                aud = np.squeeze(aud)
            self._queue.put((aud, rate), block=False)