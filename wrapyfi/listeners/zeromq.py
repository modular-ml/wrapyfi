import logging
import json
import time
import os
from typing import Optional

import numpy as np
import cv2
import zmq

from wrapyfi.connect.listeners import Listener, Listeners, ListenerWatchDog
from wrapyfi.middlewares.zeromq import ZeroMQMiddlewarePubSub
from wrapyfi.encoders import JsonDecodeHook


SOCKET_IP = os.environ.get("WRAPYFI_ZEROMQ_SOCKET_IP", "127.0.0.1")
SOCKET_PUB_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_SOCKET_PUB_PORT", 5555))
ZEROMQ_PUBSUB_MONITOR_TOPIC = os.environ.get("WRAPYFI_ZEROMQ_PUBSUB_MONITOR_TOPIC", "ZEROMQ/CONNECTIONS")
ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN = os.environ.get("WRAPYFI_ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN", "process")
WATCHDOG_POLL_REPEAT = None


class ZeroMQListener(Listener):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", should_wait: bool = True,
                 socket_ip: str = SOCKET_IP, socket_pub_port: int = SOCKET_PUB_PORT,
                 pubsub_monitor_topic: str = ZEROMQ_PUBSUB_MONITOR_TOPIC,
                 pubsub_monitor_listener_spawn: Optional[str] = ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN,
                 zeromq_kwargs: Optional[dict] = None, **kwargs):
        """
        Initialize the subscriber.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param socket_ip: str: IP address of the socket. Default is '127.0.0.1
        :param socket_pub_port: int: Port of the socket for publishing.
                                 Note that the subscriber listens directly to this port which is proxied .
                                 Default is 5555
        :param pubsub_monitor_topic: str: Topic to monitor the connections. Default is 'ZEROMQ/CONNECTIONS'
        :param pubsub_monitor_listener_spawn: str: Whether to spawn the PUB/SUB monitor listener as a process or thread. Default is 'process'
        :param zeromq_kwargs: dict: Additional kwargs for the ZeroMQ middleware
        :param kwargs: dict: Additional kwargs for the subscriber
        """
        if carrier or carrier != "tcp":
            logging.warning("[ZeroMQ] ZeroMQ does not support other carriers than TCP for PUB/SUB pattern. Using TCP.")
            carrier = "tcp"
        super().__init__(name, in_topic, carrier=carrier, should_wait=should_wait, **kwargs)

        self.socket_address = f"{carrier}://{socket_ip}:{socket_pub_port}"

        ZeroMQMiddlewarePubSub.activate(socket_pub_address=self.socket_address,
                                        pubsub_monitor_topic=pubsub_monitor_topic,
                                        pubsub_monitor_listener_spawn=pubsub_monitor_listener_spawn,
                                        **zeromq_kwargs or {})

        ZeroMQMiddlewarePubSub().shared_monitor_data.add_topic(self.in_topic)

    def await_connection(self, socket=None, in_topic: Optional[str] = None, repeats: Optional[int] = None):
        """
        Wait for the connection to be established.

        :param socket: zmq.Socket: Socket to await connection to
        :param in_topic: str: Name of the input topic
        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        connected = False
        if in_topic is None:
            in_topic = self.in_topic
        logging.info(f"[ZeroMQ] Waiting for input port: {in_topic}")
        if repeats is None:
            if self.should_wait:
                repeats = -1
            else:
                return True

        while repeats > 0 or repeats <= -1:
            repeats -= 1
            connected = ZeroMQMiddlewarePubSub().shared_monitor_data.is_connected(in_topic)
            if connected:
                logging.info(f"[ZeroMQ] Connected to input port: {in_topic}")
                break
            time.sleep(0.2)
        return connected

    def read_socket(self, socket):
        """
        Read the socket.

        :param socket: zmq.Socket: Socket to read from
        :return: bytes: Data read from the socket
        :return: yarp.Value: Value read from the port
        """
        while True:
            obj = socket.read(shouldWait=False)
            if self.should_wait and obj is None:
                time.sleep(0.005)
            else:
                return obj

    def close(self):
        """
        Close the subscriber.
        """
        ZeroMQMiddlewarePubSub().shared_monitor_data.remove_topic(self.in_topic)
        time.sleep(0.2)

        if hasattr(self, "_socket") and self._socket:
            if self._socket is not None:
                self._socket.close()

    def __del__(self):
        self.close()


@Listeners.register("NativeObject", "zeromq")
class ZeroMQNativeObjectListener(ZeroMQListener):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", should_wait: bool = True,
                 deserializer_kwargs: Optional[dict] = None, **kwargs):
        """
        The NativeObject listener using the ZeroMQ message construct assuming the data is serialized as a JSON string.
        Deserializes the data (including plugins) using the decoder and parses it to a native object.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, in_topic, carrier=carrier, should_wait=should_wait, **kwargs)
        self._socket = self._netconnect = None

        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        if not self.should_wait:
            ListenerWatchDog().add_listener(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection to the publisher.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        self._socket = zmq.Context.instance().socket(zmq.SUB)
        for socket_property in ZeroMQMiddlewarePubSub().zeromq_kwargs.items():
            if isinstance(socket_property[1], str):
                self._socket.setsockopt_string(getattr(zmq, socket_property[0]), socket_property[1])
            else:
                self._socket.setsockopt(getattr(zmq, socket_property[0]), socket_property[1])
        self._socket.connect(self.socket_address)
        self._topic = self.in_topic.encode("utf-8")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self.in_topic)

        established = self.await_connection(repeats=repeats)

        return self.check_establishment(established)

    def listen(self):
        """
        Listen for a message.

        :return: Any: The received message as a native python object
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return None
        if self._socket.poll(timeout=None if self.should_wait else 0):
            obj = self._socket.recv_multipart()
            if obj is not None:
                return json.loads(obj[1].decode(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
            else:
                return None
        else:
            return None


@Listeners.register("Image", "zeromq")
class ZeroMQImageListener(ZeroMQNativeObjectListener):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", should_wait: bool = True,
                 width: int = -1, height: int = -1, rgb: bool = True, fp: bool = False, jpg: bool = False, **kwargs):
        """
        The Image listener using the ZeroMQ message construct parsed to a numpy array.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol (e.g. 'tcp'). Default is 'tcp'
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param width: int: Width of the image. Default is -1 (use the width of the received image)
        :param height: int: Height of the image. Default is -1 (use the height of the received image)
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: bool: True if the image should be decompressed from JPG. Default is False
        """
        super().__init__(name, in_topic, carrier=carrier, should_wait=should_wait, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        self._type = np.float32 if self.fp else np.uint8

    def listen(self):
        """
        Listen for a message.

        :return: np.ndarray: The received message as a numpy array formatted as a cv2 image np.ndarray[img_height, img_width, channels]
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return None
        if self._socket.poll(timeout=None if self.should_wait else 0):
            obj = self._socket.recv_multipart()
            if obj is None:
                return None
            elif self.jpg:
                if self.rgb:
                    img = cv2.imdecode(np.frombuffer(obj[2], np.uint8), cv2.IMREAD_COLOR)
                else:
                    img = cv2.imdecode(np.frombuffer(obj[2], np.uint8), cv2.IMREAD_GRAYSCALE)
                return img
            else:
                img = json.loads(obj[2].decode(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
                if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                        not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
                    raise ValueError("Incorrect image shape for listener")
                return img
        else:
            return None


@Listeners.register("AudioChunk", "zeromq")
class ZeroMQAudioChunkListener(ZeroMQImageListener):
    def __init__(self, name: str, in_topic: str, carrier: str = "tcp", should_wait: bool = True,
                 channels: int = 1, rate: int = 44100, chunk: int = -1, **kwargs):
        """
        The AudioChunk listener using the ZeroMQ message construct parsed to a numpy array.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param queue_size: int: Size of the queue for the subscriber. Default is 5
        :param channels: int: Number of channels in the audio. Default is 1
        :param rate: int: Sampling rate of the audio. Default is 44100
        :param chunk: int: Number of samples in the audio chunk. Default is -1 (use the chunk size of the received audio)
        """
        super().__init__(name, in_topic, carrier=carrier, should_wait=should_wait,
                         width=chunk, height=channels, rgb=False, fp=True, jpg=False, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def listen(self):
        """
        Listen for a message.

        :return: Tuple[np.ndarray, int]: The received message as a numpy array formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return None
        if self._socket.poll(timeout=None if self.should_wait else 0):
            obj = self._socket.recv_multipart()
            chunk, channels, rate, aud = json.loads(obj[2].decode(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs) if obj is not None else None
            if 0 < self.rate != rate:
                raise ValueError("Incorrect audio rate for listener")
            if 0 < self.chunk != chunk or self.channels != channels or aud.size != chunk * channels:
                raise ValueError("Incorrect audio shape for listener")
            return aud, rate
        else:
            return None, self.rate


@Listeners.register("Properties", "zeromq")
class ZeroMQPropertiesListener(ZeroMQListener):
    def __init__(self, name, in_topic, **kwargs):
        super().__init__(name, in_topic, **kwargs)
        raise NotImplementedError
