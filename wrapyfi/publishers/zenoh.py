import logging
import json
import time
import os
import threading
from typing import Optional, Tuple

import numpy as np
import cv2
import zenoh

from wrapyfi.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapyfi.middlewares.zenoh import ZenohMiddlewarePubSub
from wrapyfi.encoders import JsonEncoder


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

# WARNING: unlike other publishers, Zenoh is sensitive to the polling interval and might need to be changed to avoid crashing
WATCHDOG_POLL_INTERVAL = float(os.getenv("WRAPYFI_ZENOH_RETRY_INTERVAL", 0.2))
WATCHDOG_POLL_REPEATS = int(os.getenv("WRAPYFI_ZENOH_MAX_REPEATS", -1))

# Ensure the monitor listener spawn type is compatible
if ZENOH_MONITOR_LISTENER_SPAWN == "process":
    ZENOH_MONITOR_LISTENER_SPAWN = "thread"
    logging.warning(
        "[Zenoh] Wrapyfi does not support multiprocessing for Zenoh. "
        "Switching automatically to 'thread' mode."
    )


class ZenohPublisher(Publisher):
    """
    Base Zenoh publisher class that configures and initializes Zenoh middleware.
    Sets up connection handling and establishes connection on demand.
    """

    def __init__(
        self,
        name: str,
        out_topic: str,
        should_wait: bool = True,
        ip: str = ZENOH_IP,
        port: int = ZENOH_PORT,
        mode: str = ZENOH_MODE,
        monitor_listener_spawn: Optional[str] = ZENOH_MONITOR_LISTENER_SPAWN,
        zenoh_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the Zenoh publisher.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param ip: str: IP address for the Zenoh connection. Default is '127.0.0.1'
        :param port: int: Port for the Zenoh connection. Default is 7447
        :param mode: str: Mode for Zenoh session (`peer` or `client`)
        :param monitor_listener_spawn: str: Listener spawn method (thread or process)
        :param zenoh_kwargs: dict: Additional kwargs for the Zenoh middleware
        """
        out_topic = out_topic.strip("/")
        super().__init__(name, out_topic, should_wait=should_wait, **kwargs)

        # Prepare Zenoh configuration
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

    def _prepare_config(self, zenoh_kwargs):
        """
        Converts keyword arguments to a zenoh.Config object.

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
        logging.info(f"[Zenoh] Waiting for output connection: {out_topic}")
        if repeats is None:
            repeats = -1 if self.should_wait else 1
        while repeats > 0 or repeats <= -1:
            repeats -= 1
            connected = ZenohMiddlewarePubSub._instance.is_connected()
            if connected:
                ZenohMiddlewarePubSub._instance.session.declare_publisher(out_topic)
                logging.info(f"[Zenoh] Output connection established: {out_topic}")
                break
            time.sleep(WATCHDOG_POLL_INTERVAL)
        return connected

    def close(self):
        """
        Close the publisher.
        """
        logging.info(f"[Zenoh] Closing publisher for topic: {self.out_topic}")
        ZenohMiddlewarePubSub._instance.session.close()
        time.sleep(0.2)

    def __del__(self):
        self.close()


@Publishers.register("NativeObject", "zenoh")
class ZenohNativeObjectPublisher(ZenohPublisher):
    """
    Zenoh NativeObject publisher for publishing JSON-encoded native objects using JsonEncoder.
    Serializes the data and publishes it as a JSON string.
    """

    def __init__(
        self,
        name: str,
        out_topic: str,
        should_wait: bool = True,
        multi_threaded: bool = False,
        serializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The NativeObjectPublisher using the Zenoh message construct assuming a combination of Python native objects
        and numpy arrays as input. Serializes the data (including plugins) using the encoder and sends it as a string.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g., 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param multi_threaded: bool: Whether to use a separate session for each thread. Default is False
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        """
        super().__init__(name, out_topic, should_wait=should_wait, **kwargs)
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
        established = self.await_connection(repeats=repeats)
        return self.check_establishment(established)

    def publish(self, obj):
        """
        Publish the object to the middleware.

        :param obj: object: Object to publish
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEATS)
            if not established:
                return
            else:
                time.sleep(0.2)

        obj_str = json.dumps(
            obj,
            cls=self._plugin_encoder,
            **self._plugin_kwargs,
            **self._serializer_kwargs,
        )
        ZenohMiddlewarePubSub._instance.session.put(self.out_topic, obj_str)


@Publishers.register("Image", "zenoh")
class ZenohImagePublisher(ZenohNativeObjectPublisher):
    """
    Zenoh Image publisher for publishing image data as numpy arrays.
    Supports publishing both JPEG-compressed and raw images.
    """

    def __init__(
        self,
        name: str,
        out_topic: str,
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
        The ImagePublisher using the Zenoh message construct assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g., 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param multi_threaded: bool: Whether to use a separate session for each thread. Default is False
        :param width: int: Width of the image. Default is -1 (dynamic width)
        :param height: int: Height of the image. Default is -1 (dynamic height)
        :param rgb: bool: True if the image is RGB, False if grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if integer. Default is False
        :param jpg: bool: True if the image should be compressed as JPG. Default is False
        """
        super().__init__(name, out_topic, should_wait=should_wait, **kwargs)
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
            established = self.establish(repeats=WATCHDOG_POLL_REPEATS)
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
            _, img_encoded = cv2.imencode(".jpg", img)
            img_bytes = img_encoded.tobytes()
            header = {"timestamp": time.time()}
        else:
            img_bytes = img.tobytes()
            header = {
                "timestamp": time.time(),
                "shape": img.shape,
                "dtype": str(img.dtype),
            }

        header_bytes = json.dumps(header).encode("utf-8")
        payload = header_bytes + b"\n" + img_bytes

        ZenohMiddlewarePubSub._instance.session.put(self.out_topic, payload)


@Publishers.register("AudioChunk", "zenoh")
class ZenohAudioChunkPublisher(ZenohNativeObjectPublisher):
    """
    Zenoh AudioChunk publisher for publishing audio data as numpy arrays.
    Supports publishing multi-channel audio with variable sampling rates.
    """

    def __init__(
        self,
        name: str,
        out_topic: str,
        should_wait: bool = True,
        multi_threaded: bool = False,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = -1,
        **kwargs,
    ):
        """
        The AudioChunkPublisher using the Zenoh message construct assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g., 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param multi_threaded: bool: Whether to use a separate session for each thread. Default is False
        :param channels: int: Number of channels. Default is 1
        :param rate: int: Sampling rate. Default is 44100
        :param chunk: int: Chunk size. Default is -1 (dynamic chunk size)
        """
        super().__init__(
            name,
            out_topic,
            should_wait=should_wait,
            multi_threaded=multi_threaded,
            **kwargs,
        )
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    import json

    def publish(self, aud: Tuple[np.ndarray, int]):
        """
        Publish the audio chunk to the middleware.

        :param aud: Tuple[np.ndarray, int]: Audio chunk to publish formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEATS)
            if not established:
                return
            else:
                time.sleep(0.2)

        aud_array, rate = aud
        if aud_array is None:
            return
        if 0 < self.rate != rate:
            raise ValueError("Incorrect audio rate for publisher")

        chunk, channels = (
            aud_array.shape if len(aud_array.shape) > 1 else (aud_array.shape[0], 1)
        )
        self.chunk = chunk if self.chunk == -1 else self.chunk
        self.channels = channels if self.channels == -1 else self.channels
        if 0 < self.chunk != chunk or 0 < self.channels != channels:
            raise ValueError("Incorrect audio shape for publisher")

        aud_array = np.require(aud_array, dtype=np.float32, requirements="C")
        aud_bytes = aud_array.tobytes()

        header = {
            "shape": aud_array.shape,
            "dtype": str(aud_array.dtype),
            "rate": rate,
            "timestamp": time.time(),
        }
        header_bytes = json.dumps(header).encode("utf-8")
        payload = header_bytes + b"\n" + aud_bytes

        ZenohMiddlewarePubSub._instance.session.put(self.out_topic, payload)


@Publishers.register("Properties", "zenoh")
class ZenohPropertiesPublisher(ZenohPublisher):

    def __init__(self, name, out_topic, **kwargs):
        super().__init__(name, out_topic, **kwargs)
        raise NotImplementedError
