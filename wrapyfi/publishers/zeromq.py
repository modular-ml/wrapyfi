import logging
import json
import time
import os
import threading
import base64
import io
from typing import Optional, Tuple

import numpy as np
import cv2
import zmq

from wrapyfi.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapyfi.middlewares.zeromq import ZeroMQMiddlewarePubSub
from wrapyfi.encoders import JsonEncoder


SOCKET_IP = os.environ.get("WRAPYFI_ZEROMQ_SOCKET_IP", "127.0.0.1")
SOCKET_PUB_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_SOCKET_PUB_PORT", 5555))
SOCKET_SUB_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_SOCKET_SUB_PORT", 5556))
PARAM_PUB_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_PARAM_PUB_PORT", 5655))
PARAM_SUB_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_PARAM_SUB_PORT", 5656))
PARAM_REQREP_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_PARAM_REQREP_PORT", 5659))
PARAM_POLL_INTERVAL = int(os.environ.get("WRAPYFI_ZEROMQ_PARAM_POLL_INTERVAL", 1))
START_PROXY_BROKER = (
    os.environ.get("WRAPYFI_ZEROMQ_START_PROXY_BROKER", True) != "False"
)
PROXY_BROKER_SPAWN = os.environ.get("WRAPYFI_ZEROMQ_PROXY_BROKER_SPAWN", "process")
ZEROMQ_PUBSUB_MONITOR_TOPIC = os.environ.get(
    "WRAPYFI_ZEROMQ_PUBSUB_MONITOR_TOPIC", "ZEROMQ/CONNECTIONS"
)
ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN = os.environ.get(
    "WRAPYFI_ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN", "process"
)
WATCHDOG_POLL_REPEAT = None


if os.name == "nt" and PROXY_BROKER_SPAWN == "process" and START_PROXY_BROKER:
    PROXY_BROKER_SPAWN = "thread"
    logging.warning(
        "[ZeroMQ] Wrapyfi does not support multiprocessing on Windows. Please set "
        "the environment variable WRAPYFI_ZEROMQ_PROXY_BROKER_SPAWN='thread'. "
        "Switching automatically to 'thread' mode. "
    )

if os.name == "nt" and ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN == "process":
    ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN = "thread"
    logging.warning(
        "[ZeroMQ] Wrapyfi does not support multiprocessing on Windows. Please set "
        "the environment variable WRAPYFI_ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN='thread'. "
        "Switching automatically to 'thread' mode. "
    )


class ZeroMQPublisher(Publisher):
    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: str = "tcp",
        should_wait: bool = True,
        socket_ip: str = SOCKET_IP,
        socket_pub_port: int = SOCKET_PUB_PORT,
        socket_sub_port: int = SOCKET_SUB_PORT,
        start_proxy_broker: bool = START_PROXY_BROKER,
        proxy_broker_spawn: str = PROXY_BROKER_SPAWN,
        pubsub_monitor_topic: str = ZEROMQ_PUBSUB_MONITOR_TOPIC,
        pubsub_monitor_listener_spawn: Optional[
            str
        ] = ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN,
        zeromq_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the publisher and start the proxy broker if necessary.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param socket_ip: str: IP address of the socket. Default is '127.0.0.1'
        :param socket_pub_port: int: Port of the socket for publishing. Default is 5555
        :param socket_sub_port: int: Port of the socket for subscribing. Default is 5556
        :param start_proxy_broker: bool: Whether to start a proxy broker. Default is True
        :param proxy_broker_spawn: str: Whether to spawn the proxy broker as a process or thread. Default is 'process'
        :param pubsub_monitor_topic: str: Topic to monitor the connections. Default is 'ZEROMQ/CONNECTIONS'
        :param pubsub_monitor_listener_spawn: str: Whether to spawn the PUB/SUB monitor listener as a process or thread. Default is 'process'
        :param zeromq_kwargs: dict: Additional kwargs for the ZeroMQ Pub/Sub middleware
        :param kwargs: Additional kwargs for the publisher
        """
        if carrier or carrier != "tcp":
            logging.warning(
                "[ZeroMQ] ZeroMQ does not support other carriers than TCP for PUB/SUB pattern. Using TCP."
            )
            carrier = "tcp"
        super().__init__(
            name, out_topic, carrier=carrier, should_wait=should_wait, **kwargs
        )

        self.socket_pub_address = f"{carrier}://{socket_ip}:{socket_pub_port}"
        self.socket_sub_address = f"{carrier}://{socket_ip}:{socket_sub_port}"

        ZeroMQMiddlewarePubSub.activate(
            socket_pub_address=self.socket_pub_address,
            socket_sub_address=self.socket_sub_address,
            start_proxy_broker=start_proxy_broker,
            proxy_broker_spawn=proxy_broker_spawn,
            pubsub_monitor_topic=pubsub_monitor_topic,
            pubsub_monitor_listener_spawn=pubsub_monitor_listener_spawn,
            **zeromq_kwargs or {},
        )

        ZeroMQMiddlewarePubSub().shared_monitor_data.add_topic(self.out_topic)

    def await_connection(
        self, out_topic: Optional[str] = None, repeats: Optional[int] = None
    ):
        """
        Wait for the connection to be established.

        :param out_topic: str: Name of the output topic
        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        connected = False
        if out_topic is None:
            out_topic = self.out_topic
        logging.info(f"[ZeroMQ] Waiting for output connection: {out_topic}")
        if repeats is None:
            repeats = -1 if self.should_wait else 1
        while repeats > 0 or repeats <= -1:
            repeats -= 1
            # allowing should_wait into the loop for consistency with other publishers only
            connected = (
                ZeroMQMiddlewarePubSub().shared_monitor_data.is_connected(out_topic)
                or not self.should_wait
            )
            if connected:
                logging.info(f"[ZeroMQ] Output connection established: {out_topic}")
                break
            time.sleep(0.02)
        return connected

    def close(self):
        """
        Close the publisher.
        """
        ZeroMQMiddlewarePubSub().shared_monitor_data.remove_topic(self.out_topic)
        time.sleep(0.2)

        if hasattr(self, "_socket") and self._socket:
            if self._socket is not None:
                self._socket.close()

    def __del__(self):
        self.close()


@Publishers.register("NativeObject", "zeromq")
class ZeroMQNativeObjectPublisher(ZeroMQPublisher):

    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: str = "tcp",
        should_wait: bool = True,
        multi_threaded: bool = False,
        serializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The NativeObjectPublisher using the ZeroMQ message construct assuming a combination of python native objects
        and numpy arrays as input. Serializes the data (including plugins) using the encoder and sends it as a string.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param multi_threaded: bool: Whether to use a separate socket for each thread. Default is False
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        :param kwargs: dict: Additional kwargs for the publisher
        """
        super().__init__(
            name, out_topic, carrier=carrier, should_wait=should_wait, **kwargs
        )
        self._socket = self._netconnect = None
        if multi_threaded:
            self._thread_local_storage = threading.local()

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection to the publisher.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        self._socket = zmq.Context.instance().socket(zmq.PUB)
        for socket_property in ZeroMQMiddlewarePubSub().zeromq_kwargs.items():
            if isinstance(socket_property[1], str):
                self._socket.setsockopt_string(
                    getattr(zmq, socket_property[0]), socket_property[1]
                )
            else:
                self._socket.setsockopt(
                    getattr(zmq, socket_property[0]), socket_property[1]
                )
        self._socket.connect(self.socket_sub_address)
        self._topic = self.out_topic.encode()
        established = self.await_connection(repeats=repeats)
        return self.check_establishment(established)

    def _get_socket(self):
        """
        Retrieve or create a ZeroMQ socket specific to the current thread.
        """
        if not hasattr(self._thread_local_storage, "socket"):
            # Initialize a new socket for the thread
            self._thread_local_storage.socket = zmq.Context.instance().socket(zmq.PUB)
            for socket_property in ZeroMQMiddlewarePubSub().zeromq_kwargs.items():
                if isinstance(socket_property[1], str):
                    self._thread_local_storage.socket.setsockopt_string(
                        getattr(zmq, socket_property[0]), socket_property[1]
                    )
                else:
                    self._thread_local_storage.socket.setsockopt(
                        getattr(zmq, socket_property[0]), socket_property[1]
                    )
            self._thread_local_storage.socket.connect(self.socket_sub_address)
        return self._thread_local_storage.socket

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
        if hasattr(self, "_thread_local_storage"):
            socket = self._get_socket()
        else:
            socket = self._socket
        socket.send_multipart([self._topic, obj_str.encode()])


@Publishers.register("Image", "zeromq")
class ZeroMQImagePublisher(ZeroMQNativeObjectPublisher):

    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: str = "tcp",
        should_wait: bool = True,
        multi_threaded: bool = False,
        width: int = -1,
        height: int = -1,
        rgb: bool = True,
        fp: bool = False,
        jpg: bool = False,
        **kwargs,
    ):
        """
        The ImagePublisher using the ZeroMQ message construct assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param multi_threaded: bool: Whether to use a separate socket for each thread. Default is False
        :param width: int: Width of the image. Default is -1 meaning that the width is not fixed
        :param height: int: Height of the image. Default is -1 meaning that the height is not fixed
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: bool: True if the image should be compressed as JPG. Default is False
        """
        super().__init__(
            name, out_topic, carrier=carrier, should_wait=should_wait, **kwargs
        )
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        self._type = np.float32 if self.fp else np.uint8

    def publish(self, img: np.ndarray):
        """
        Publish the image to the middleware.

        :param img: np.ndarray: Image to publish formatted as a cv2 image np.ndarray[img_height, img_width, channels]
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
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)

        if self.jpg:
            img_str = np.array(cv2.imencode(".jpg", img)[1]).tostring()
        else:
            img_str = json.dumps(
                img,
                cls=self._plugin_encoder,
                **self._plugin_kwargs,
                serializer_kwrags=self._serializer_kwargs,
            ).encode()
        img_header = "{timestamp:" + str(time.time()) + "}"
        if hasattr(self, "_thread_local_storage"):
            socket = self._get_socket()
        else:
            socket = self._socket
        socket.send_multipart([self._topic, img_header.encode(), img_str])


@Publishers.register("AudioChunk", "zeromq")
class ZeroMQAudioChunkPublisher(ZeroMQNativeObjectPublisher):
    def __init__(
        self,
        name: str,
        out_topic: str,
        carrier: str = "tcp",
        should_wait: bool = True,
        multi_threaded: bool = False,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = -1,
        **kwargs,
    ):
        """
        The AudioChunkPublisher using the ZeroMQ message construct assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param multi_threaded: bool: Whether to use a separate socket for each thread. Default is False
        :param channels: int: Number of channels. Default is 1
        :param rate: int: Sampling rate. Default is 44100
        :param chunk: int: Chunk size. Default is -1 meaning that the chunk size is not fixed
        """
        super().__init__(
            name,
            out_topic,
            carrier=carrier,
            should_wait=should_wait,
            multi_threaded=multi_threaded,
            width=chunk,
            height=channels,
            rgb=False,
            fp=True,
            jpg=False,
            **kwargs,
        )
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def publish(self, aud: Tuple[np.ndarray, int]):
        """
        Publish the audio chunk to the middleware.

        :param aud: Tuple[np.ndarray, int]: Audio chunk to publish formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
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

        aud_str = json.dumps(
            (chunk, channels, rate, aud),
            cls=self._plugin_encoder,
            **self._plugin_kwargs,
            serializer_kwrags=self._serializer_kwargs,
        ).encode()
        aud_header = "{timestamp:" + str(time.time()) + "}"
        if hasattr(self, "_thread_local_storage"):
            socket = self._get_socket()
        else:
            socket = self._socket
        socket.send_multipart([self._topic, aud_header.encode(), aud_str])


@Publishers.register("Properties", "zeromq")
class ZeroMQPropertiesPublisher(ZeroMQPublisher):

    def __init__(self, name, out_topic, **kwargs):
        super().__init__(name, out_topic, **kwargs)
        raise NotImplementedError
