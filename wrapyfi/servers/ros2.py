import logging
import sys
import json
import threading
import queue
from typing import Optional, Tuple, Union

import numpy as np
import rclpy
from rclpy.node import Node

from wrapyfi.connect.servers import Server, Servers
from wrapyfi.middlewares.ros2 import ROS2Middleware
from wrapyfi.utils.serialization_encoders import JsonEncoder, JsonDecodeHook
from wrapyfi.utils.image_encoders import JpegEncoder


class ROS2Server(Server, Node):

    def __init__(
        self, name: str, out_topic: str, ros2_kwargs: Optional[dict] = None, **kwargs
    ):
        """
        Initialize the server.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param ros2_kwargs: dict: Additional kwargs for the ROS 2 middleware
        :param kwargs: dict: Additional kwargs for the server
        """
        carrier = "tcp"
        if "carrier" in kwargs and kwargs["carrier"] not in ["", None]:
            logging.warning(
                "[ROS 2] ROS 2 currently does not support explicit carrier setting for REQ/REP pattern. Using TCP."
            )
        if "carrier" in kwargs:
            del kwargs["carrier"]

        ROS2Middleware.activate(**ros2_kwargs or {})
        Server.__init__(self, name, out_topic, carrier=carrier, **kwargs)
        Node.__init__(self, name + str(hex(id(self))))

    def close(self):
        """
        Close the server.
        """
        if hasattr(self, "_server") and self._server:
            if self._server is not None:
                self.destroy_node()
        if hasattr(self, "_background_callback") and self._background_callback:
            if self._background_callback is not None:
                self._background_callback.join()

    def __del__(self):
        self.close()


@Servers.register("NativeObject", "ros2")
class ROS2NativeObjectServer(ROS2Server):
    SEND_QUEUE = queue.Queue(maxsize=1)
    RECEIVE_QUEUE = queue.Queue(maxsize=1)

    def __init__(
        self,
        name: str,
        out_topic: str,
        serializer_kwargs: Optional[dict] = None,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Specific server handling native Python objects, serializing them to JSON strings for transmission.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, out_topic, **kwargs)
        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._server = None

    def establish(self):
        """
        Establish the connection to the server
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

        self._server = self.create_service(
            ROS2NativeObjectService, self.out_topic, self._service_callback
        )

        self._req_msg = ROS2NativeObjectService.Request()
        self._rep_msg = ROS2NativeObjectService.Response()
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
            self._background_callback = threading.Thread(
                name="ros2_server", target=rclpy.spin_once, args=(self,), kwargs={}
            )
            self._background_callback.setDaemon(True)
            self._background_callback.start()

            request = ROS2NativeObjectServer.RECEIVE_QUEUE.get(block=True)
            [args, kwargs] = json.loads(
                request.request,
                object_hook=self._plugin_decoder_hook,
                **self._deserializer_kwargs,
            )
            return args, kwargs
        except Exception as e:
            logging.error("[ROS 2] Service call failed %s" % e)
            return [], {}

    @staticmethod
    def _service_callback(request, _response):
        """
        Callback for the ROS 2 service.

        :param request: ROS2NativeObjectService.Request: The request message
        :param _response: ROS2NativeObjectService.Response: The response message
        """
        ROS2NativeObjectServer.RECEIVE_QUEUE.put(request)
        return ROS2NativeObjectServer.SEND_QUEUE.get(block=True)

    def reply(self, obj):
        """
        Serialize the provided Python object to a JSON string and send it as a reply to the client.
        The method uses the configured JSON encoder for serialization before sending the resultant string to the client.

        :param obj: Any: The Python object to be serialized and sent
        """
        try:
            obj_str = json.dumps(
                obj,
                cls=self._plugin_encoder,
                **self._plugin_kwargs,
                serializer_kwrags=self._serializer_kwargs,
            )
            self._rep_msg.response = obj_str
            ROS2NativeObjectServer.SEND_QUEUE.put(self._rep_msg, block=False)
        except queue.Full:
            logging.warning(
                f"[ROS 2] Discarding data because queue is full. "
                f"This happened due to bad synchronization in {self.__name__}"
            )


@Servers.register("Image", "ros2")
class ROS2ImageServer(ROS2Server):
    SEND_QUEUE = queue.Queue(maxsize=1)
    RECEIVE_QUEUE = queue.Queue(maxsize=1)

    def __init__(
        self,
        name: str,
        out_topic: str,
        width: int = -1,
        height: int = -1,
        rgb: bool = True,
        fp: bool = False,
        jpg: Union[bool, dict] = False,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Specific server handling native Python objects, serializing them to JSON strings for transmission.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param width: int: Width of the image. Default is -1 (use the width of the received image)
        :param height: int: Height of the image. Default is -1 (use the height of the received image)
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
        :param jpg: Union[bool, dict]: If True, compress as JPG with default settings. If a dict, pass arguments to JpegEncoder. Default is False
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, out_topic, **kwargs)
        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

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
        if self.jpg:
            self._encoding = "jpeg"
            self._type = np.uint8
            self._image_encoder = JpegEncoder(
                **(self.jpg if isinstance(self.jpg, dict) else {})
            )

        self._server = None

    def establish(self):
        """
        Establish the connection to the server.
        """
        try:
            from wrapyfi_ros2_interfaces.srv import (
                ROS2ImageService,
                ROS2CompressedImageService,
            )
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
        if self.jpg:
            self._server = self.create_service(
                ROS2CompressedImageService, self.out_topic, self._service_callback
            )
            self._req_msg = ROS2CompressedImageService.Request()
            self._rep_msg = ROS2CompressedImageService.Response()
        else:
            self._server = self.create_service(
                ROS2ImageService, self.out_topic, self._service_callback
            )
            self._req_msg = ROS2ImageService.Request()
            self._rep_msg = ROS2ImageService.Response()
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
            self._background_callback = threading.Thread(
                name="ros2_server", target=rclpy.spin_once, args=(self,), kwargs={}
            )
            self._background_callback.setDaemon(True)
            self._background_callback.start()

            request = ROS2ImageServer.RECEIVE_QUEUE.get(block=True)
            [args, kwargs] = json.loads(
                request.request,
                object_hook=self._plugin_decoder_hook,
                **self._deserializer_kwargs,
            )
            return args, kwargs
        except Exception as e:
            logging.error("[ROS 2] Service call failed %s" % e)
            return [], {}

    @staticmethod
    def _service_callback(request, _response):
        """
        Callback for the ROS 2 service.

        :param request: ROS2NativeObjectService.Request: The request message
        :param _response: ROS2NativeObjectService.Response: The response message
        :return: ROS2NativeObjectService.Response: The response message
        """
        ROS2ImageServer.RECEIVE_QUEUE.put(request)
        return ROS2ImageServer.SEND_QUEUE.get(block=True)

    def reply(self, img: np.ndarray):
        """
        Serialize the provided image data and send it as a reply to the client.

        :param img: np.ndarray: Image to send formatted as a cv2 image - np.ndarray[img_height, img_width, channels]
        """
        try:
            if (
                0 < self.width != img.shape[1]
                or 0 < self.height != img.shape[0]
                or not (
                    (img.ndim == 2 and not self.rgb)
                    or (img.ndim == 3 and self.rgb and img.shape[2] == 3)
                )
            ):
                raise ValueError("Incorrect image shape for publisher")
            img = np.require(img, dtype=self._type, requirements="C")
            img_msg = self._rep_msg.response
            if self.jpg:
                img_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
                img_msg.format = "jpeg"
                img_msg.data = self._image_encoder.encode_jpg_image(
                    img, return_numpy=True
                ).tobytes()
            else:
                img_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
                img_msg.height = img.shape[0]
                img_msg.width = img.shape[1]
                img_msg.encoding = self._encoding
                img_msg.is_bigendian = img.dtype.byteorder == ">" or (
                    img.dtype.byteorder == "=" and sys.byteorder == "big"
                )
                img_msg.step = img.strides[0]
                img_msg.data = img.tobytes()
            self._rep_msg.response = img_msg
            ROS2ImageServer.SEND_QUEUE.put(self._rep_msg, block=False)
        except queue.Full:
            logging.warning(
                f"[ROS 2] Discarding data because queue is full. "
                f"This happened due to bad synchronization in {self.__name__}"
            )


@Servers.register("AudioChunk", "ros2")
class ROS2AudioChunkServer(ROS2Server):
    SEND_QUEUE = queue.Queue(maxsize=1)
    RECEIVE_QUEUE = queue.Queue(maxsize=1)

    def __init__(
        self,
        name: str,
        out_topic: str,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = -1,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Specific server handling audio data as numpy arrays.

        :param name: str: Name of the server
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param channels: int: Number of channels in the audio. Default is 1
        :param rate: int: Sampling rate of the audio. Default is 44100
        :param chunk: int: Number of samples in the audio chunk. Default is -1 (use the chunk size of the received audio)
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, out_topic, **kwargs)
        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def establish(self):
        """
        Establish the connection to the server.
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
        self._server = self.create_service(
            ROS2AudioService, self.out_topic, self._service_callback
        )
        self._req_msg = ROS2AudioService.Request()
        self._rep_msg = ROS2AudioService.Response()
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
            self._background_callback = threading.Thread(
                name="ros2_server", target=rclpy.spin_once, args=(self,), kwargs={}
            )
            self._background_callback.setDaemon(True)
            self._background_callback.start()

            request = ROS2AudioChunkServer.RECEIVE_QUEUE.get(block=True)
            [args, kwargs] = json.loads(
                request.request,
                object_hook=self._plugin_decoder_hook,
                **self._deserializer_kwargs,
            )
            return args, kwargs
        except Exception as e:
            logging.error("[ROS 2] Service call failed %s" % e)
            return [], {}

    @staticmethod
    def _service_callback(request, _response):
        """
        Callback for the ROS 2 service.

        :param request: ROS2NativeObjectService.Request: The request message
        :param _response: ROS2NativeObjectService.Response: The response message
        :return: ROS2NativeObjectService.Response: The response message
        """
        ROS2AudioChunkServer.RECEIVE_QUEUE.put(request)
        return ROS2AudioChunkServer.SEND_QUEUE.get(block=True)

    def reply(self, aud: Tuple[np.ndarray, int]):
        """
        Serialize the provided audio data and send it as a reply to the client.

        :param aud: Tuple[np.ndarray, int]: Audio chunk to publish formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        try:
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

            aud_msg = self._rep_msg.response
            aud_msg.header.stamp = self.get_clock().now().to_msg()
            aud_msg.chunk_size = chunk
            aud_msg.channels = channels
            aud_msg.sample_rate = rate
            aud_msg.is_bigendian = aud.dtype.byteorder == ">" or (
                aud.dtype.byteorder == "=" and sys.byteorder == "big"
            )
            aud_msg.encoding = "S16BE" if aud_msg.is_bigendian else "S16LE"
            aud_msg.step = aud.strides[0]
            aud_msg.data = aud.tobytes()  # (aud * 32767.0).tobytes()
            self._rep_msg.response = aud_msg
            ROS2AudioChunkServer.SEND_QUEUE.put(self._rep_msg, block=False)
        except queue.Full:
            logging.warning(
                f"[ROS 2] Discarding data because queue is full. "
                f"This happened due to bad synchronization in {self.__name__}"
            )
