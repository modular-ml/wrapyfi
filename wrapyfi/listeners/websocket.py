import logging
import json
import time
import os
import queue
from typing import Optional

import numpy as np
import cv2

from wrapyfi.connect.listeners import Listener, Listeners, ListenerWatchDog
from wrapyfi.middlewares.websocket import WebSocketMiddlewarePubSub
from wrapyfi.encoders import JsonDecodeHook

SOCKET_IP = os.environ.get("WRAPYFI_WEBSOCKET_SOCKET_IP", "127.0.0.1")
SOCKET_PORT = int(os.environ.get("WRAPYFI_WEBSOCKET_SOCKET_PORT", 5000))
WEBSOCKET_NAMESPACE = os.environ.get("WRAPYFI_WEBSOCKET_NAMESPACE", "/")
WEBSOCKET_MONITOR_LISTENER_SPAWN = os.environ.get(
    "WRAPYFI_WEBSOCKET_MONITOR_LISTENER_SPAWN", "thread"
)
WATCHDOG_POLL_REPEAT = None
if WEBSOCKET_MONITOR_LISTENER_SPAWN == "process":
    WEBSOCKET_MONITOR_LISTENER_SPAWN = "thread"
    logging.warning(
        "[WebSocket] Wrapyfi does not support multiprocessing for Websockets. Please set "
        "the environment variable WRAPYFI_WEBSOCKET_MONITOR_LISTENER_SPAWN='thread'. "
        "Switching automatically to 'thread' mode."
    )


class WebSocketListener(Listener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        socket_ip: str = SOCKET_IP,
        socket_port: int = SOCKET_PORT,
        namespace: str = WEBSOCKET_NAMESPACE,
        monitor_listener_spawn: Optional[str] = WEBSOCKET_MONITOR_LISTENER_SPAWN,
        websocket_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the subscriber.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic (e.g., 'topic')
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param socket_ip: str: IP address of the socket. Default is '127.0.0.1'
        :param socket_port: int: Port of the socket for publishing. Default is 5000
        :param namespace: str: Namespace of the WebSocket. Default is '/'
        :param monitor_listener_spawn: str: Whether to spawn the monitor listener as a process or thread. Default is 'thread'
        :param websocket_kwargs: dict: Additional kwargs for the WebSocket middleware
        :param kwargs: dict: Additional kwargs for the subscriber
        """
        super().__init__(name, in_topic, should_wait=should_wait, **kwargs)

        self.socket_address = f"http://{socket_ip}:{socket_port}{namespace}"

        # Activate the WebSocket middleware
        WebSocketMiddlewarePubSub.activate(
            socket_address=self.socket_address,
            monitor_listener_spawn=monitor_listener_spawn,
            **(websocket_kwargs or {}),
        )

        # Register the callback for the topic
        WebSocketMiddlewarePubSub._instance.register_callback(self.in_topic, self.on_message)

        if not self.should_wait:
            ListenerWatchDog().add_listener(self)

    def await_connection(
        self, in_topic: Optional[str] = None, repeats: Optional[int] = None
    ):
        """
        Wait until the WebSocket connection is established.

        :param in_topic: str: The topic to monitor for connection
        :param repeats: int: The number of times to check for the connection, None for infinite.
        """
        if in_topic is None:
            in_topic = self.in_topic
        logging.info(f"[WebSocket] Waiting for input port: {in_topic}")
        if repeats is None:
            repeats = -1 if self.should_wait else 0

        # Ensure to call is_connected() on the singleton instance
        while repeats > 0 or repeats == -1:
            if repeats != -1:
                repeats -= 1
            connected = WebSocketMiddlewarePubSub._instance.is_connected()  # Use the instance
            logging.debug(f"Connection status: {connected}")
            if connected:
                logging.info(f"[WebSocket] Connected to input port: {in_topic}")
                return True
            time.sleep(0.2)
        return False

    def close(self):
        """
        Close the subscriber.
        """
        # You might want to deregister the callback or perform other cleanup here
        # However, since the middleware handles cleanup on exit, we might not need to do anything
        pass

    def __del__(self):
        self.close()


@Listeners.register("NativeObject", "websocket")
class WebSocketNativeObjectListener(WebSocketListener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The NativeObject listener using the WebSocket message construct assuming the data is serialized as a JSON string.
        Deserializes the data (including plugins) using the decoder and parses it to a native object.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic (e.g., 'topic')
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, in_topic, should_wait=should_wait, **kwargs)
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}
        self._message_queue = queue.Queue()

    def on_message(self, data):
        try:
            obj = json.loads(
                data,
                object_hook=self._plugin_decoder_hook,
                **self._deserializer_kwargs,
            )
            self._message_queue.put(obj)
            logging.debug(f"Message queued for topic {self.in_topic}: {obj}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from topic {self.in_topic}: {e}")

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection to the publisher.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        established = self.await_connection(repeats=repeats)
        return self.check_establishment(established)

    def listen(self):
        """
        Listen for a message.

        :return: Any: The received message as a native Python object
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return None

        try:
            obj = self._message_queue.get(block=self.should_wait)
            return obj
        except queue.Empty:
            return None


@Listeners.register("Image", "websocket")
class WebSocketImageListener(WebSocketListener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        width: int = -1,
        height: int = -1,
        rgb: bool = True,
        fp: bool = False,
        jpg: bool = False,
        **kwargs,
    ):
        """
        The Image listener using the WebSocket message construct parsed to a numpy array.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic (e.g., 'topic')
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param width: int: Width of the image. Default is -1 (use the width of the received image)
        :param height: int: Height of the image. Default is -1 (use the height of the received image)
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: bool: True if the image should be decompressed from JPG. Default is False
        """
        super().__init__(name, in_topic, should_wait=should_wait, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        self._type = np.float32 if self.fp else np.uint8
        self._message_queue = queue.Queue()

    def on_message(self, data):
        if self.jpg:
            img_bytes = data.get('image_bytes', None)
            if img_bytes is not None:
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                if self.rgb:
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                else:
                    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                self._message_queue.put(img)
        else:
            img_bytes = data.get('image_bytes', None)
            shape = data.get('shape', None)
            dtype = data.get('dtype', None)
            if img_bytes is not None and shape is not None and dtype is not None:
                img_array = np.frombuffer(img_bytes, dtype=dtype)
                img = img_array.reshape(shape)
                self._message_queue.put(img)

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection to the publisher.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        return super().establish(repeats=repeats, **kwargs)

    def listen(self):
        """
        Listen for a message.

        :return: np.ndarray: The received image as a numpy array formatted as a cv2 image np.ndarray[img_height, img_width, channels]
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return None

        try:
            img = self._message_queue.get(block=self.should_wait)
            if (
                (self.width > 0 and self.width != img.shape[1]) or
                (self.height > 0 and self.height != img.shape[0]) or
                not (
                    (img.ndim == 2 and not self.rgb) or
                    (img.ndim == 3 and self.rgb and img.shape[2] == 3)
                )
            ):
                raise ValueError("Incorrect image shape for listener")
            return img
        except queue.Empty:
            return None


@Listeners.register("AudioChunk", "websocket")
class WebSocketAudioChunkListener(WebSocketListener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = -1,
        **kwargs,
    ):
        """
        The AudioChunk listener using the WebSocket message construct parsed to a numpy array.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic (e.g., 'topic')
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param channels: int: Number of channels in the audio. Default is 1
        :param rate: int: Sampling rate of the audio. Default is 44100
        :param chunk: int: Number of samples in the audio chunk. Default is -1 (use the chunk size of the received audio)
        """
        super().__init__(name, in_topic, should_wait=should_wait, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._message_queue = queue.Queue()

    def on_message(self, data):
        chunk = data.get('chunk')
        channels = data.get('channels')
        rate = data.get('rate')
        aud = data.get('aud')

        if 0 < self.rate != rate:
            raise ValueError("Incorrect audio rate for listener")
        if (
            (0 < self.chunk != chunk) or
            self.channels != channels or
            len(aud) != chunk * channels
        ):
            raise ValueError("Incorrect audio shape for listener")
        aud = np.array(aud, dtype=np.float32).reshape((chunk, channels))
        self._message_queue.put((aud, rate))

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection to the publisher.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        return super().establish(repeats=repeats, **kwargs)

    def listen(self):
        """
        Listen for a message.

        :return: Tuple[np.ndarray, int]: The received audio chunk as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return None, self.rate

        try:
            aud, rate = self._message_queue.get(block=self.should_wait)
            return aud, rate
        except queue.Empty:
            return None, self.rate


@Listeners.register("Properties", "websocket")
class WebSocketPropertiesListener(WebSocketListener):
    def __init__(self, name, in_topic, **kwargs):
        super().__init__(name, in_topic, **kwargs)
        raise NotImplementedError
