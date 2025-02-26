import logging
import json
import time
import os
import base64

import numpy as np
from typing import Optional, Tuple, Union
from socketio import exceptions

from wrapyfi.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapyfi.middlewares.websocket import WebSocketMiddlewarePubSub
from wrapyfi.utils.serialization_encoders import JsonEncoder
from wrapyfi.utils.image_encoders import JpegEncoder


SOCKET_IP = os.environ.get("WRAPYFI_WEBSOCKET_SOCKET_IP", "127.0.0.1")
SOCKET_PORT = int(os.environ.get("WRAPYFI_WEBSOCKET_SOCKET_PORT", 5000))
WEBSOCKET_NAMESPACE = os.environ.get("WRAPYFI_WEBSOCKET_NAMESPACE", "/")

WATCHDOG_POLL_REPEAT = None


class WebSocketPublisher(Publisher):

    def __init__(
        self,
        name: str,
        out_topic: str,
        should_wait: bool = True,
        socket_ip: str = SOCKET_IP,
        socket_port: int = SOCKET_PORT,
        namespace: str = WEBSOCKET_NAMESPACE,
        websocket_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the publisher.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g. 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param socket_ip: str: IP address of the socket. Default is '127.0.0.1'
        :param socket_port: int: Port of the socket for publishing. Default is 5000
        :param namespace: str: Namespace of the WebSocket. Default is '/'
        :param websocket_kwargs: dict: Additional kwargs for the WebSocket middleware
        :param kwargs: Additional kwargs for the publisher
        """
        super().__init__(name, out_topic, should_wait=should_wait, **kwargs)

        self.socket_address = f"http://{socket_ip}:{socket_port}{namespace}"

        WebSocketMiddlewarePubSub.activate(
            socket_address=self.socket_address,
            **(websocket_kwargs or {}),
        )

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
        logging.info(f"[WebSocket] Waiting for output connection: {out_topic}")
        if repeats is None:
            repeats = -1 if self.should_wait else 1

        while repeats > 0 or repeats <= -1:
            repeats -= 1
            connected = WebSocketMiddlewarePubSub._instance.is_connected()
            if connected:
                logging.info(f"[WebSocket] Output connection established: {out_topic}")
                break
            time.sleep(0.02)
        return connected

    def close(self):
        """
        Close the publisher.
        """
        logging.info(f"[WebSocket] Closing publisher for topic: {self.out_topic}")
        WebSocketMiddlewarePubSub._instance.socketio_client.disconnect()
        time.sleep(0.2)

    def __del__(self):
        self.close()


@Publishers.register("NativeObject", "websocket")
class WebSocketNativeObjectPublisher(WebSocketPublisher):

    def __init__(
        self,
        name: str,
        out_topic: str,
        should_wait: bool = True,
        serializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The NativeObjectPublisher using the WebSocket message construct assuming a combination of python native objects
        and numpy arrays as input. Serializes the data (including plugins) using the encoder and sends it as a string.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g. 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        :param kwargs: dict: Additional kwargs for the publisher
        """
        super().__init__(name, out_topic, should_wait=should_wait, **kwargs)

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
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
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
        socketio_client = WebSocketMiddlewarePubSub._instance.socketio_client
        try:
            socketio_client.emit(self.out_topic, obj_str)
        except (exceptions.BadNamespaceError, exceptions.DisconnectedError):
            self.established = False


@Publishers.register("Image", "websocket")
class WebSocketImagePublisher(WebSocketNativeObjectPublisher):

    def __init__(
        self,
        name: str,
        out_topic: str,
        should_wait: bool = True,
        width: int = -1,
        height: int = -1,
        rgb: bool = True,
        fp: bool = False,
        jpg: Union[bool, dict] = False,
        **kwargs,
    ):
        """
        The ImagePublisher using the WebSocket message construct assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g. 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param width: int: Width of the image. Default is -1 meaning that the width is not fixed
        :param height: int: Height of the image. Default is -1 meaning that the height is not fixed
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: Union[bool, dict]: If True, compress as JPG with default settings. If a dict, pass arguments to JpegEncoder. Default is False
        """
        super().__init__(name, out_topic, should_wait=should_wait, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        self._type = np.float32 if self.fp else np.uint8

        if self.jpg:
            self._image_encoder = JpegEncoder(
                **(self.jpg if isinstance(self.jpg, dict) else {})
            )

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

        socketio_client = WebSocketMiddlewarePubSub._instance.socketio_client

        if self.jpg:
            img_bytes = self._image_encoder.encode_jpg_image(img)
            header = {"timestamp": time.time()}
        else:
            img_bytes = img.tobytes()
            header = {
                "timestamp": time.time(),
                "shape": img.shape,
                "dtype": str(img.dtype),
            }

        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        try:
            socketio_client.emit(self.out_topic, [header, img_base64])
        except (exceptions.BadNamespaceError, exceptions.DisconnectedError):
            self.established = False


@Publishers.register("AudioChunk", "websocket")
class WebSocketAudioChunkPublisher(WebSocketNativeObjectPublisher):
    def __init__(
        self,
        name: str,
        out_topic: str,
        should_wait: bool = True,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = -1,
        **kwargs,
    ):
        """
        The AudioChunkPublisher using the WebSocket message construct assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g. 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param channels: int: Number of channels. Default is 1
        :param rate: int: Sampling rate. Default is 44100
        :param chunk: int: Chunk size. Default is -1 meaning that the chunk size is not fixed
        """
        super().__init__(
            name,
            out_topic,
            should_wait=should_wait,
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

        socketio_client = WebSocketMiddlewarePubSub._instance.socketio_client

        aud_bytes = aud_array.tobytes()
        header = {
            "shape": aud_array.shape,
            "dtype": str(aud_array.dtype),
            "rate": rate,
            "timestamp": time.time(),
        }

        try:
            socketio_client.emit(self.out_topic, [header, aud_bytes])
        except (exceptions.BadNamespaceError, exceptions.DisconnectedError):
            self.established = False


@Publishers.register("Properties", "websocket")
class WebSocketPropertiesPublisher(WebSocketPublisher):

    def __init__(self, name, out_topic, **kwargs):
        super().__init__(name, out_topic, **kwargs)
        raise NotImplementedError
