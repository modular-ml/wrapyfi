import logging
import sys
import json
import queue
from typing import Optional, Any, Union

import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from wrapyfi.connect.clients import Client, Clients
from wrapyfi.middlewares.ros2 import ROS2Middleware
from wrapyfi.utils.serialization_encoders import JsonEncoder, JsonDecodeHook


class ROS2Client(Client, Node):
    def __init__(
        self, name: str, in_topic: str, ros2_kwargs: Optional[dict] = None, **kwargs
    ):
        """
        Initialize the client.

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param ros2_kwargs: dict: Additional kwargs for the ROS 2 middleware
        :param kwargs: dict: Additional kwargs for the client
        """
        carrier = "tcp"
        if "carrier" in kwargs and kwargs["carrier"] not in ["", None]:
            logging.warning(
                "[ROS 2] ROS 2 currently does not support explicit carrier setting for PUB/SUB pattern. Using TCP."
            )
        if "carrier" in kwargs:
            del kwargs["carrier"]
        ROS2Middleware.activate(**ros2_kwargs or {})
        Client.__init__(self, name, in_topic, carrier=carrier, **kwargs)
        Node.__init__(self, name + str(hex(id(self))))

    def close(self):
        """
        Close the client.
        """
        if hasattr(self, "_client") and self._client:
            if self._client is not None:
                self.destroy_node()

    def __del__(self):
        self.close()


@Clients.register("NativeObject", "ros2")
class ROS2NativeObjectClient(ROS2Client):
    def __init__(
        self,
        name: str,
        in_topic: str,
        serializer_kwargs: Optional[dict] = None,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The NativeObject listener using the ROS 2 String message assuming the data is serialized as a JSON string.
        Deserializes the data (including plugins) using the decoder and parses it to a Python object.

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, in_topic, **kwargs)
        self._client = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

    def establish(self):
        """
        Establish the client's connection to the ROS 2 service.
        """
        try:
            from wrapyfi_ros2_interfaces.srv import ROS2NativeObjectService
        except ImportError:
            import wrapyfi

            logging.error(
                "[ROS 2] Could not import ROS2NativeObjectService. "
                "Make sure the ROS 2 services in wrapyfi_extensions/wrapyfi_ros2_interfaces are compiled. "
                "Refer to the documentation for more information: \n"
                + wrapyfi.__doc__
                + "ros2_interfaces_lnk.html"
            )
            sys.exit(1)
        self._client = self.create_client(ROS2NativeObjectService, self.in_topic)
        self._req_msg = ROS2NativeObjectService.Request()

        while not self._client.wait_for_service(timeout_sec=1.0):
            logging.info("[ROS 2] Service not available, waiting again...")
        self.established = True

    def request(self, *args, **kwargs):
        """
        Send a request to the ROS 2 service.

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        :return: Any: The response from the ROS 2 service
        """
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except Exception as e:
            logging.error("[ROS 2] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        """
        Internal method to send a request to the ROS 2 service.

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        """
        # transmit args to server
        args_str = json.dumps(
            [args, kwargs],
            cls=self._plugin_encoder,
            **self._plugin_kwargs,
            serializer_kwrags=self._serializer_kwargs,
        )
        self._req_msg.request = args_str
        future = self._client.call_async(self._req_msg)
        # receive message from server
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    msg = future.result()
                    obj = json.loads(
                        msg.response,
                        object_hook=self._plugin_decoder_hook,
                        **self._deserializer_kwargs,
                    )
                    self._queue.put(obj, block=False)
                except Exception as e:
                    logging.error("[ROS 2] Service call failed: %s" % e)
                break

    def _await_reply(self) -> Any:
        """
        Wait for and return the reply from the ROS 2 service.

        :return: Any: The response from the ROS 2 service
        """
        try:
            reply = self._queue.get(block=True)
            return reply
        except queue.Full:
            logging.warning(
                f"[ROS 2] Discarding data because queue is full. "
                f"This happened due to bad synchronization in {self.__name__}"
            )
            return None


@Clients.register("Image", "ros2")
class ROS2ImageClient(ROS2Client):
    def __init__(
        self,
        name: str,
        in_topic: str,
        width: int = -1,
        height: int = -1,
        rgb: bool = True,
        fp: bool = False,
        jpg: Union[bool, dict] = False,
        serializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The Image client using the ROS 2 Image message parsed to a numpy array.

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param width: int: The width of the image. Default is -1 (use the width of the received image)
        :param height: int: The height of the image. Default is -1 (use the height of the received image)
        :param rgb: bool: Whether the image is RGB. Default is True
        :param fp: bool: Whether to utilize floating-point precision. Default is False
        :param jpg: bool: True if the image should be decompressed from JPG. Default is False
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        """
        super().__init__(name, in_topic, **kwargs)
        self._client = None
        self._queue = queue.Queue(maxsize=1)

        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        if self.fp:
            self._encoding = "32FC3" if self.rgb else "32FC1"
            self._type = np.float32
        else:
            self._encoding = "bgr8" if self.rgb else "mono8"
            self._type = np.uint8
        self._pixel_bytes = (3 if self.rgb else 1) * np.dtype(self._type).itemsize

        self._client = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}

    def establish(self):
        """
        Establish the client's connection to the ROS 2 service.
        """
        try:
            from wrapyfi_ros2_interfaces.srv import (
                ROS2ImageService,
                ROS2CompressedImageService,
            )
        except ImportError:
            import wrapyfi

            logging.error(
                "[ROS 2] Could not import ROS2ImageService. "
                "Make sure the ROS 2 services in wrapyfi_extensions/wrapyfi_ros2_interfaces are compiled. "
                "Refer to the documentation for more information: \n"
                + wrapyfi.__doc__
                + "ros2_interfaces_lnk.html"
            )
            sys.exit(1)
        if self.jpg:
            self._client = self.create_client(ROS2CompressedImageService, self.in_topic)
            self._req_msg = ROS2CompressedImageService.Request()
        else:
            self._client = self.create_client(ROS2ImageService, self.in_topic)
            self._req_msg = ROS2ImageService.Request()

        while not self._client.wait_for_service(timeout_sec=1.0):
            logging.info("[ROS 2] Service not available, waiting again...")
        self.established = True

    def request(self, *args, **kwargs):
        """
        Send a request to the ROS 2 service.

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        :return: Any: The response from the ROS 2 service
        """
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except Exception as e:
            logging.error("[ROS 2] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        """
        Internal method to send a request to the ROS 2 service.

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        """
        # transmit args to server
        args_str = json.dumps(
            [args, kwargs],
            cls=self._plugin_encoder,
            **self._plugin_kwargs,
            serializer_kwrags=self._serializer_kwargs,
        )
        self._req_msg.request = args_str
        future = self._client.call_async(self._req_msg)
        # receive message from server
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    msg = future.result()
                    data = msg.response
                    if self.jpg:
                        self._queue.put((data.format, data.data), block=False)
                    else:
                        self._queue.put(
                            (
                                data.height,
                                data.width,
                                data.encoding,
                                data.is_bigendian,
                                data.data,
                            ),
                            block=False,
                        )
                except Exception as e:
                    logging.error("[ROS 2] Service call failed: %s" % e)
                break

    def _await_reply(self):
        """
        Wait for and return the reply from the ROS 2 service.

        :return: np.array: The received image from the ROS 2 service
        """
        try:
            if self.jpg:
                format, data = self._queue.get(block=True)
                if format != "jpeg":
                    raise ValueError(f"Unsupported image format: {format}")
                if self.rgb:
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                else:
                    img = cv2.imdecode(
                        np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE
                    )
            else:
                height, width, encoding, is_bigendian, data = self._queue.get(
                    block=True
                )
                if encoding != self._encoding:
                    raise ValueError("Incorrect encoding for listener")
                if (
                    0 < self.width != width
                    or 0 < self.height != height
                    or len(data) != height * width * self._pixel_bytes
                ):
                    raise ValueError("Incorrect image shape for listener")
                img = np.frombuffer(
                    data,
                    dtype=np.dtype(self._type).newbyteorder(
                        ">" if is_bigendian else "<"
                    ),
                ).reshape((height, width, -1))
                if img.shape[2] == 1:
                    img = img.squeeze(axis=2)
            return img
        except queue.Full:
            logging.warning(
                f"[ROS 2] Discarding data because queue is full. "
                f"This happened due to bad synchronization in {self.__name__}"
            )
            return None


@Clients.register("AudioChunk", "ros2")
class ROS2AudioChunkClient(ROS2Client):
    def __init__(
        self,
        name: str,
        in_topic: str,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = -1,
        serializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The AudioChunk client using the ROS 2 Audio message parsed to a numpy array.

        :param name: str: Name of the client
        :param in_topic: str: Name of the input topic preceded by '/' (e.g. '/topic')
        :param channels: int: Number of channels in the audio. Default is 1
        :param rate: int: Sampling rate of the audio. Default is 44100
        :param chunk: int: Number of samples in the audio chunk. Default is -1 (use the chunk size of the received audio)
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        """
        super().__init__(name, in_topic, **kwargs)
        self._client = None
        self._queue = queue.Queue(maxsize=1)

        self.channels = channels
        self.rate = rate
        self.chunk = chunk

        self._client = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}

    def establish(self):
        """
        Establish the client's connection to the ROS 2 service.
        """
        try:
            from wrapyfi_ros2_interfaces.srv import ROS2AudioService
        except ImportError:
            import wrapyfi

            logging.error(
                "[ROS 2] Could not import ROS2AudioService. "
                "Make sure the ROS 2 services in wrapyfi_extensions/wrapyfi_ros2_interfaces are compiled. "
                "Refer to the documentation for more information: \n"
                + wrapyfi.__doc__
                + "ros2_interfaces_lnk.html"
            )
            sys.exit(1)
        self._client = self.create_client(ROS2AudioService, self.in_topic)
        self._req_msg = ROS2AudioService.Request()

        while not self._client.wait_for_service(timeout_sec=1.0):
            logging.info("[ROS 2] Service not available, waiting again...")
        self.established = True

    def request(self, *args, **kwargs):
        """
        Send a request to the ROS 2 service.

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        :return: Any: The response from the ROS 2 service
        """
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except Exception as e:
            logging.error("[ROS 2] Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        """
        Internal method to send a request to the ROS 2 service.

        :param args: tuple: Positional arguments to send in the request
        :param kwargs: dict: Keyword arguments to send in the request
        """
        args_str = json.dumps(
            [args, kwargs],
            cls=self._plugin_encoder,
            **self._plugin_kwargs,
            serializer_kwrags=self._serializer_kwargs,
        )
        self._req_msg.request = args_str
        future = self._client.call_async(self._req_msg)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    msg = future.result()
                    data = msg.response
                    self._queue.put(
                        (
                            data.chunk_size,
                            data.channels,
                            data.sample_rate,
                            data.encoding,
                            data.is_bigendian,
                            data.data,
                        ),
                        block=False,
                    )
                except Exception as e:
                    logging.error("[ROS 2] Service call failed: %s" % e)
                break

    def _await_reply(self):
        """
        Wait for and return the reply from the ROS 2 service.

        :return: Tuple[np.ndarray, int]: The received message as a numpy array formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        try:
            chunk, channels, rate, encoding, is_bigendian, data = self._queue.get(
                block=False
            )
            if 0 < self.rate != rate:
                raise ValueError("Incorrect audio rate for publisher")
            if encoding not in ["S16LE", "S16BE"]:
                raise ValueError("Incorrect encoding for listener")
            if (
                0 < self.chunk != chunk
                or self.channels != channels
                or len(data) != chunk * channels * 4
            ):
                raise ValueError("Incorrect audio shape for listener")
            aud = np.frombuffer(
                data,
                dtype=np.dtype(np.float32).newbyteorder(">" if is_bigendian else "<"),
            ).reshape((chunk, channels))
            # aud = aud / 32767.0
            return aud, rate
        except queue.Full:
            logging.warning(
                f"[ROS 2] Discarding data because queue is full. "
                f"This happened due to bad synchronization in {self.__name__}"
            )
            return None
