import logging
import json
import queue
import time
import os
from typing import Optional, Union

import numpy as np
import cv2
import zenoh

from wrapyfi.connect.listeners import Listener, Listeners, ListenerWatchDog
from wrapyfi.middlewares.zenoh import ZenohMiddlewarePubSub
from wrapyfi.utils.serialization_encoders import JsonDecodeHook


# Capture environment variables for Zenoh configuration
ZENOH_IP = os.getenv("WRAPYFI_ZENOH_IP", "127.0.0.1")
ZENOH_PORT = int(os.getenv("WRAPYFI_ZENOH_PORT", 7447))
ZENOH_MODE = os.getenv("WRAPYFI_ZENOH_MODE", "peer")
ZENOH_CONNECT = json.loads(os.getenv("WRAPYFI_ZENOH_CONNECT", "[]"))
ZENOH_LISTEN = json.loads(os.getenv("WRAPYFI_ZENOH_LISTEN", "[]"))
ZENOH_CONFIG_FILEPATH = os.getenv("WRAPYFI_ZENOH_CONFIG_FILEPATH", None)
ZENOH_MONITOR_LISTENER_SPAWN = os.getenv(
    "WRAPYFI_ZENOH_MONITOR_LISTENER_SPAWN", "thread"
)

WATCHDOG_POLL_INTERVAL = float(os.getenv("WRAPYFI_ZENOH_RETRY_INTERVAL", 0.2))
WATCHDOG_POLL_REPEATS = int(os.getenv("WRAPYFI_ZENOH_MAX_REPEATS", -1))

# Ensure the monitor listener spawn type is compatible
if ZENOH_MONITOR_LISTENER_SPAWN == "process":
    ZENOH_MONITOR_LISTENER_SPAWN = "thread"
    logging.warning(
        "[Zenoh] Wrapyfi does not support multiprocessing for Zenoh. "
        "Switching automatically to 'thread' mode."
    )


class ZenohListener(Listener):
    """
    Base Zenoh listener class that configures and initializes Zenoh middleware.
    Merges listener-specific settings and environment configurations, and awaits connection.
    """

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        ip: str = ZENOH_IP,
        port: int = ZENOH_PORT,
        mode: str = ZENOH_MODE,
        zenoh_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initializes the Zenoh listener with environment or parameter-based configurations
        and waits for connection if specified.

        :param name: str: Name of the listener
        :param in_topic: str: Topic name
        :param should_wait: bool: Whether to block until a message is received
        :param ip: str: IP address for the Zenoh connection. Default is '127.0.0.1'
        :param port: int: Port for the Zenoh connection. Default is 7447
        :param mode: str: Mode for Zenoh session (`peer` or `client`)
        :param zenoh_kwargs: dict: Additional Zenoh configuration options, overridden by env variables
        :param kwargs: dict: Additional options for the listener
        """

        # Zenoh does not accept trailing or leading slashes in topic names
        in_topic = in_topic.strip("/")
        super().__init__(name, in_topic, should_wait=should_wait, **kwargs)

        # Prepare Zenoh configuration from environment variables and kwargs
        self.zenoh_config = {
            "mode": mode,
            "connect/endpoints": (
                ZENOH_CONNECT
                if isinstance(ZENOH_CONNECT, list) and ZENOH_CONNECT
                else (
                    ZENOH_CONNECT.split(",")
                    if isinstance(ZENOH_CONNECT, str)
                    else [f"tcp/{ip}:{port}"]
                )
            ),
            **(zenoh_kwargs or {}),
        }
        if ZENOH_LISTEN:
            self.zenoh_config["listen/endpoints"] = (
                ZENOH_LISTEN
                if isinstance(ZENOH_LISTEN, list)
                else ZENOH_LISTEN.split(",")
            )

        ZenohMiddlewarePubSub.activate(
            config=self._prepare_config(self.zenoh_config), **kwargs
        )

        self.established = False

    def _prepare_config(self, zenoh_kwargs):
        """
        Converts keyword arguments to a zenoh.Config object and merges with environment-based settings.

        :param zenoh_kwargs: dict: Configuration parameters
        :return: zenoh.Config: Configured Zenoh session
        """
        config = (
            zenoh.Config().from_file(ZENOH_CONFIG_FILEPATH)
            if ZENOH_CONFIG_FILEPATH
            else zenoh.Config()
        )
        for key, value in zenoh_kwargs.items():
            config.insert_json5(key, json.dumps(value))
        return config

    def await_connection(
        self, in_topic: Optional[str] = None, repeats: int = WATCHDOG_POLL_REPEATS
    ):
        """
        Waits for the Zenoh connection to be established.

        :param in_topic: str: Topic name for connection
        :param repeats: int: Number of retry attempts
        :return: bool: True if connection is established, False otherwise
        """
        in_topic = in_topic or self.in_topic
        while repeats != 0:
            repeats -= 1 if repeats > 0 else 0
            if ZenohMiddlewarePubSub._instance.is_connected():
                logging.info(f"[Zenoh] Connected to topic {in_topic}")
                return True
            logging.debug(f"Waiting for connection on topic {in_topic}")
            time.sleep(WATCHDOG_POLL_INTERVAL)
        return False

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection to the publisher.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        established = self.await_connection(repeats=repeats)
        established = self.check_establishment(established)
        if established:
            ZenohMiddlewarePubSub._instance.register_callback(
                self.in_topic, self.on_message
            )
        return established

    def close(self):
        """
        Closes the Zenoh listener.
        This can be overridden by child classes to add cleanup operations.
        """
        pass


@Listeners.register("NativeObject", "zenoh")
class ZenohNativeObjectListener(ZenohListener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Zenoh NativeObject listener for handling JSON-encoded native objects.
        Decodes incoming messages to native Python objects using JsonDecodeHook.

        :param name: str: Name of the listener
        :param in_topic: str: Name of the input topic
        :param should_wait: bool: Whether to wait for messages
        :param deserializer_kwargs: dict: Keyword arguments for the JSON deserializer
        """
        super().__init__(name, in_topic, should_wait=should_wait, **kwargs)
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._message_queue = queue.Queue()
        self._deserializer_kwargs = deserializer_kwargs or {}

        if not self.should_wait:
            ListenerWatchDog().add_listener(self)

    def on_message(self, sample):
        """
        Handles incoming messages by decoding JSON into native objects using JsonDecodeHook.

        :param sample: zenoh.Sample: The Zenoh sample received
        """
        try:
            obj = json.loads(
                sample.payload.to_bytes(),
                object_hook=self._plugin_decoder_hook,
                **self._deserializer_kwargs,
            )
            self._message_queue.put(obj)
            logging.debug(f"Queued message for topic {self.in_topic}: {obj}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from topic {self.in_topic}: {e}")

    def listen(self):
        """
        Listen for a message, ensuring the connection is established.

        :return: Any: The received message as a native Python object
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEATS)
            if not established:
                return None

        try:
            return self._message_queue.get(block=self.should_wait)
        except queue.Empty:
            return None


@Listeners.register("Image", "zenoh")
class ZenohImageListener(ZenohNativeObjectListener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        width: int = -1,
        height: int = -1,
        rgb: bool = True,
        jpg: Union[bool, dict] = False,
        **kwargs,
    ):
        """
        Zenoh Image listener for handling image messages.
        Converts incoming data to OpenCV images, supporting JPEG and raw formats.

        :param name: str: Name of the listener
        :param in_topic: str: Name of the input topic
        :param should_wait: bool: Whether to wait for messages
        :param width: int: Expected image width, -1 to use received width
        :param height: int: Expected image height, -1 to use received height
        :param rgb: bool: True if the image is RGB, False if grayscale
        :param jpg: bool: True if the image is JPEG-compressed
        """
        super().__init__(name, in_topic, should_wait=should_wait, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.jpg = jpg
        self._message_queue = queue.Queue()

    def on_message(self, sample):
        """
        Handles incoming image messages, converting data to OpenCV format.

        :param sample: zenoh.Sample: Zenoh sample payload
        """
        try:
            # Split payload into header and image data
            payload = sample.payload.to_bytes()
            header_bytes, img_bytes = payload.split(
                b"\n", 1
            )  # Split at the first newline

            header = json.loads(header_bytes.decode("utf-8"))
            np_data = np.frombuffer(img_bytes, dtype=np.uint8)

            if self.jpg:
                img = cv2.imdecode(
                    np_data, cv2.IMREAD_COLOR if self.rgb else cv2.IMREAD_GRAYSCALE
                )
            else:
                shape = header.get(
                    "shape", (self.height, self.width, 3 if self.rgb else 1)
                )
                img = np_data.reshape(shape)

            # Place the decoded image into the message queue
            self._message_queue.put(img)
        except Exception as e:
            logging.error(f"Failed to process image message: {e}")

    def listen(self):
        """
        Listen for a message, ensuring the connection is established.

        :return: np.ndarray: The received image as an OpenCV-formatted array
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEATS)
            if not established:
                return None

        try:
            return self._message_queue.get(block=self.should_wait)
        except queue.Empty:
            return None


@Listeners.register("AudioChunk", "zenoh")
class ZenohAudioChunkListener(ZenohNativeObjectListener):

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
        Zenoh AudioChunk listener for handling audio messages.
        Converts incoming data to numpy arrays for audio processing.

        :param name: str: Name of the listener
        :param in_topic: str: Name of the input topic
        :param should_wait: bool: Whether to wait for messages
        :param channels: int: Number of audio channels
        :param rate: int: Sampling rate of the audio
        :param chunk: int: Number of samples in the audio chunk
        """
        super().__init__(name, in_topic, should_wait=should_wait, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self._message_queue = queue.Queue()

    def on_message(self, sample):
        """
        Processes incoming audio messages into structured numpy arrays.

        :param sample: zenoh.Sample: Zenoh sample payload
        """
        try:
            payload = sample.payload.to_bytes()
            header_bytes, aud_bytes = payload.split(b"\n", 1)
            header = json.loads(header_bytes.decode("utf-8"))
            shape = header.get("shape")
            rate = header.get("rate")
            aud_array = np.frombuffer(aud_bytes, dtype=np.float32).reshape(shape)
            self._message_queue.put((aud_array, rate))
        except Exception as e:
            logging.error(f"Failed to process audio message: {e}")

    def listen(self):
        """
        Listen for a message, ensuring the connection is established.

        :return: Tuple[np.ndarray, int]: The received audio chunk and sampling rate
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEATS)
            if not established:
                return None, self.rate

        try:
            return self._message_queue.get(block=self.should_wait)
        except queue.Empty:
            return None, self.rate
