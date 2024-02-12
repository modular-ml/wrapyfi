import logging
import json
import queue
import os
from typing import Optional

import numpy as np
import cv2
import zmq

from wrapyfi.connect.clients import Client, Clients
from wrapyfi.middlewares.zeromq import ZeroMQMiddlewareReqRep
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook

SOCKET_IP = os.environ.get("WRAPYFI_ZEROMQ_SOCKET_IP", "127.0.0.1")
SOCKET_REP_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_SOCKET_REP_PORT", 5558))
WATCHDOG_POLL_REPEAT = None


class ZeroMQClient(Client):
    def __init__(
        self,
        name,
        in_topic,
        carrier="tcp",
        socket_ip: str = SOCKET_IP,
        socket_rep_port: int = SOCKET_REP_PORT,
        zeromq_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the client.

        :param name: str: Name of the client
        :param out_topic: str: Topics are not supported for the REQ/REP pattern in ZeroMQ. Any given topic is ignored
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param zeromq_kwargs: dict: Additional kwargs for the ZeroMQ middleware
        :param kwargs: dict: Additional kwargs for the client
        """
        if in_topic != "":
            logging.warning(
                f"[ZeroMQ] ZeroMQ does not support topics for the REQ/REP pattern. Topic {in_topic} removed"
            )
            in_topic = ""
        if carrier or carrier != "tcp":
            logging.warning(
                "[ZeroMQ] ZeroMQ does not support other carriers than TCP for REQ/REP pattern. Using TCP."
            )
            carrier = "tcp"
        super().__init__(name, in_topic, carrier=carrier, **kwargs)

        self.socket_address = f"{carrier}://{socket_ip}:{socket_rep_port}"

        ZeroMQMiddlewareReqRep.activate(**zeromq_kwargs or {})

    def close(self):
        """
        Close the subscriber.
        """
        if hasattr(self, "_socket") and self._socket:
            if self._socket is not None:
                self._socket.close()

    def __del__(self):
        self.close()


@Clients.register("NativeObject", "zeromq")
class ZeroMQNativeObjectClient(ZeroMQClient):
    def __init__(
        self,
        name: str,
        in_topic: str,
        carrier: str = "tcp",
        serializer_kwargs: Optional[dict] = None,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Specific client for handling native Python objects, serializing them to JSON strings for transmission.

        :param name: str: Name of the client
        :param in_topic: str: Topics are not supported for the REQ/REP pattern in ZeroMQ. Any given topic is ignored
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for REQ/REP pattern. Default is 'tcp'
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, in_topic, carrier=carrier, **kwargs)

        self._plugin_encoder = JsonEncoder
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._queue = queue.Queue(maxsize=1)

        self.establish()

    def establish(self, **kwargs):
        """
        Establish the connection to the server.
        """
        self._socket = zmq.Context().instance().socket(zmq.REQ)
        for socket_property in ZeroMQMiddlewareReqRep().zeromq_kwargs.items():
            if isinstance(socket_property[1], str):
                self._socket.setsockopt_string(
                    getattr(zmq, socket_property[0]), socket_property[1]
                )
            else:
                self._socket.setsockopt(
                    getattr(zmq, socket_property[0]), socket_property[1]
                )
        self._socket.connect(self.socket_address)

        self.established = True

    def request(self, *args, **kwargs):
        """
        Serialize the provided Python objects to JSON strings, send a request to the server, and await a reply.
        The method uses the configured JSON encoder for serialization before sending the resultant string to the server.

        :param args: tuple: Arguments to be sent to the server
        :param kwargs: dict: Keyword arguments to be sent to the server
        :return: Any: The Python object received from the server, deserialized using the configured JSON decoder hook
        """
        try:
            self._request(*args, **kwargs)
        except zmq.ZMQError as e:
            logging.error("[ZeroMQ] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        """
        Internal method to serialize the request, send it to the server, and receive the reply.

        :param args: tuple: Arguments to be serialized and sent
        :param kwargs: dict: Keyword arguments to be serialized and sent
        """
        args_str = json.dumps(
            [args, kwargs], cls=self._plugin_encoder, **self._serializer_kwargs
        )
        self._socket.send_string(args_str)

        obj_str = self._socket.recv_string()
        obj = json.loads(
            obj_str, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs
        )
        self._queue.put(obj, block=False)

    def _await_reply(self):
        """
        Internal method to retrieve the reply from the server from the queue and return it.

        :return: Any: The Python object received from the server, deserialized using the configured JSON decoder hook
        """
        try:
            reply = self._queue.get(block=True)
            return reply
        except queue.Empty:
            logging.warning(
                f"[ZeroMQ] Discarding data because queue is empty. "
                f"This happened due to bad synchronization in {self.__class__.__name__}"
            )
            return None


@Clients.register("Image", "zeromq")
class ZeroMQImageClient(ZeroMQNativeObjectClient):
    def __init__(
        self,
        name: str,
        in_topic: str,
        carrier: str = "tcp",
        width: int = -1,
        height: int = -1,
        rgb: bool = True,
        fp: bool = False,
        jpg: bool = False,
        serializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The Image client using the ZeroMQ message construct parsed to a numpy array.

        :param name: str: Name of the client
        :param in_topic: str: Topics are not supported for the REQ/REP pattern in ZeroMQ. Any given topic is ignored
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param width: int: Width of the image. Default is -1 (use the width of the received image)
        :param height: int: Height of the image. Default is -1 (use the height of the received image)
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: bool: True if the image should be compressed to JPG before sending. Default is False
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        """
        super().__init__(
            name,
            in_topic,
            carrier=carrier,
            serializer_kwargs=serializer_kwargs,
            **kwargs,
        )
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        self._type = np.float32 if self.fp else np.uint8

    def _request(self, *args, **kwargs):
        """
        Internal method to serialize the request, send it to the server, and receive the reply.

        :param args: tuple: Arguments to be serialized and sent
        :param kwargs: dict: Keyword arguments to be serialized and sent
        """
        args_str = json.dumps(
            [args, kwargs], cls=self._plugin_encoder, **self._serializer_kwargs
        )
        self._socket.send_string(args_str)

        if self.jpg:
            reply_bytes = self._socket.recv()
            reply_img = cv2.imdecode(
                np.frombuffer(reply_bytes, np.uint8), cv2.IMREAD_ANYCOLOR
            )
        else:
            reply_str = self._socket.recv_string()
            reply_img_list = json.loads(reply_str)
            reply_img = np.array(reply_img_list["img"], dtype=self._type)
        self._queue.put(reply_img, block=False)

    def _await_reply(self):
        """
        Internal method to retrieve the reply from the server from the queue and return it.

        :return: np.ndarray: The image received from the server as a NumPy array
        """
        try:
            img = self._queue.get(block=True)
            height, width, channels = img.shape
            if (
                0 < self.width != width
                or 0 < self.height != height
                or img.size != height * width * (3 if self.rgb else 1)
            ):
                raise ValueError("Incorrect image shape for subscriber")
            return img
        except queue.Empty:
            logging.warning(
                f"[ZeroMQ] Discarding data because queue is empty. "
                f"This happened due to bad synchronization in {self.__class__.__name__}"
            )
            return None


@Clients.register("AudioChunk", "zeromq")
class ZeroMQAudioChunkClient(ZeroMQNativeObjectClient):
    def __init__(
        self,
        name: str,
        in_topic: str,
        carrier: str = "tcp",
        channels: int = 1,
        rate: int = 44100,
        chunk: int = -1,
        serializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The AudioChunk client using the ZeroMQ message construct parsed to a numpy array.

        :param name: str: Name of the client
        :param in_topic: str: Topics are not supported for the REQ/REP pattern in ZeroMQ. Any given topic is ignored
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param width: int: Width of the image. Default is -1 (use the width of the received image)
        :param height: int: Height of the image. Default is -1 (use the height of the received image)
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: bool: True if the image should be compressed to JPG before sending. Default is False
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        """
        super().__init__(
            name,
            in_topic,
            carrier=carrier,
            serializer_kwargs=serializer_kwargs,
            **kwargs,
        )
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def _request(self, *args, **kwargs):
        """
        Internal method to serialize the request, send it to the server, and receive the reply.

        :param args: tuple: Arguments to be serialized and sent
        :param kwargs: dict: Keyword arguments to be serialized and sent
        """
        args_str = json.dumps(
            [args, kwargs], cls=self._plugin_encoder, **self._serializer_kwargs
        )
        self._socket.send_string(args_str)

        reply_str = self._socket.recv_string()
        reply_aud_list = json.loads(reply_str)
        chunk, channels, rate, aud = reply_aud_list["aud"]
        reply_aud = np.array(aud, dtype=np.float32)
        self._queue.put((chunk, channels, rate, reply_aud), block=False)

    def _await_reply(self):
        """
        Internal method to retrieve the reply from the server from the queue and return it.

        :return: Tuple[np.ndarray, int]: Audio chunk received formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        try:
            chunk, channels, rate, aud = self._queue.get(block=True)
            if 0 < self.rate != rate:
                raise ValueError("Incorrect audio rate for listener")
            if (
                0 < self.chunk != chunk
                or self.channels != channels
                or aud.size != chunk * channels
            ):
                raise ValueError("Incorrect audio shape for listener")
            return aud, rate
        except queue.Empty:
            logging.warning(
                f"[ZeroMQ] Discarding data because queue is empty. "
                f"This happened due to bad synchronization in {self.__class__.__name__}"
            )
            return None
