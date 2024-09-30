import logging
import json
import time
import os
import threading
from typing import Optional, Tuple
import random
import struct

import numpy as np
import cv2

from wrapyfi.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapyfi.middlewares.mqtt import MqttMiddlewarePubSub
from wrapyfi.encoders import JsonEncoder

MQTT_BROKER_ADDRESS = os.environ.get("WRAPYFI_MQTT_BROKER_ADDRESS", "broker.emqx.io")
MQTT_BROKER_PORT = int(os.environ.get("WRAPYFI_MQTT_BROKER_PORT", 1883))


class MqttPublisher(Publisher):

    def __init__(
        self,
        name: str,
        out_topic: str,
        should_wait: bool = True,
        broker_address: str = MQTT_BROKER_ADDRESS,
        broker_port: int = MQTT_BROKER_PORT,
        mqtt_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the publisher.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g. 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param broker_address: str: Address of the MQTT broker. Default is 'broker.emqx.io'
        :param broker_port: int: Port of the MQTT broker. Default is 1883
        :param mqtt_kwargs: dict: Additional kwargs for the MQTT middleware
        :param kwargs: dict: Additional kwargs for the publisher
        """
        super().__init__(name, out_topic, should_wait=should_wait, **kwargs)

        if mqtt_kwargs is None:
            mqtt_kwargs = {}

        if "client_id" not in mqtt_kwargs:
            mqtt_kwargs["client_id"] = f"client-{name}-{random.randint(0, 100000)}"
        # if "clean_session" not in mqtt_kwargs:
        #     mqtt_kwargs["clean_session"] = False

        # Activate the MQTT middleware
        MqttMiddlewarePubSub.activate(
            broker_address=broker_address, broker_port=broker_port, **(mqtt_kwargs or {})
        )

    def await_connection(self, out_topic: Optional[str] = None, repeats: Optional[int] = None):
        """
        Wait for the connection to be established.

        :param out_topic: str: Name of the output topic
        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        if out_topic is None:
            out_topic = self.out_topic
        logging.info(f"[MQTT] Waiting for output connection: {out_topic}")
        if repeats is None:
            repeats = -1 if self.should_wait else 0

        while repeats > 0 or repeats == -1:
            if repeats != -1:
                repeats -= 1
            connected = MqttMiddlewarePubSub._instance.is_connected()
            if connected:
                logging.info(f"[MQTT] Output connection established: {out_topic}")
                return True
            time.sleep(0.02)
        return False

    def close(self):
        """
        Close the publisher.
        """
        logging.info(f"[MQTT] Closing publisher for topic: {self.out_topic}")
        MqttMiddlewarePubSub._instance.mqtt_client.disconnect()
        time.sleep(0.2)

    def __del__(self):
        self.close()


@Publishers.register("NativeObject", "mqtt")
class MqttNativeObjectPublisher(MqttPublisher):

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
        The NativeObjectPublisher using the MQTT message construct assuming a combination of python native objects
        and numpy arrays as input. Serializes the data (including plugins) using the encoder and sends it as a string.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g. 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param multi_threaded: bool: Whether to use a separate client for each thread. Default is False
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        :param kwargs: dict: Additional kwargs for the publisher
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
            established = self.establish(repeats=10)
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
        MqttMiddlewarePubSub._instance.mqtt_client.publish(self.out_topic, obj_str)


@Publishers.register("Image", "mqtt")
class MqttImagePublisher(MqttNativeObjectPublisher):

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
        The ImagePublisher using the MQTT message construct assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g. 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param multi_threaded: bool: Whether to use a separate client for each thread. Default is False
        :param width: int: Width of the image. Default is -1 meaning that the width is not fixed
        :param height: int: Height of the image. Default is -1 meaning that the height is not fixed
        :param rgb: bool: True if the image is RGB, False if it is grayscale. Default is True
        :param fp: bool: True if the image is floating point, False if it is integer. Default is False
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
            established = self.establish(repeats=10)
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
            # Encode image as JPEG and get raw bytes
            img_bytes = cv2.imencode(".jpg", img)[1].tobytes()
            header = {
                "timestamp": time.time(),
            }
        else:
            # Convert the image to raw bytes
            img_bytes = img.tobytes()
            header = {
                "shape": img.shape,
                "dtype": str(img.dtype),
                "timestamp": time.time(),
            }

        # Serialize header to JSON and encode to bytes
        header_json = json.dumps(header)
        header_bytes = header_json.encode('utf-8')
        header_length = len(header_bytes)
        # Pack header length into 4 bytes (big-endian)
        header_length_packed = struct.pack('!I', header_length)

        # Construct the payload: header length + header bytes + image bytes
        payload = header_length_packed + header_bytes + img_bytes

        # Publish the binary payload
        MqttMiddlewarePubSub._instance.mqtt_client.publish(self.out_topic, payload)


@Publishers.register("AudioChunk", "mqtt")
class MqttAudioChunkPublisher(MqttNativeObjectPublisher):
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
        The AudioChunkPublisher using the MQTT message construct assuming a numpy array as input.

        :param name: str: Name of the publisher
        :param out_topic: str: Name of the output topic (e.g. 'topic')
        :param should_wait: bool: Whether to wait for at least one listener before unblocking the script. Default is True
        :param multi_threaded: bool: Whether to use a separate client for each thread. Default is False
        :param channels: int: Number of channels. Default is 1
        :param rate: int: Sampling rate. Default is 44100
        :param chunk: int: Chunk size. Default is -1 meaning that the chunk size is not fixed
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

    import struct
    import json
    import numpy as np
    import time
    import logging

    def publish(self, aud: Tuple[np.ndarray, int]):
        """
        Publish the audio chunk to the middleware.

        :param aud: Tuple[np.ndarray, int]: Audio chunk to publish formatted as (np.ndarray[audio_chunk, channels], int[samplerate])
        """
        if not self.established:
            established = self.establish(repeats=10)
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

        # Create the header
        header = {
            "shape": aud_array.shape,
            "dtype": str(aud_array.dtype),
            "rate": rate,
            "timestamp": time.time(),
        }

        # Serialize header to JSON and encode to bytes
        header_json = json.dumps(header)
        header_bytes = header_json.encode('utf-8')
        header_length = len(header_bytes)
        # Pack header length into 4 bytes (big-endian)
        header_length_packed = struct.pack('!I', header_length)

        # Get audio bytes
        aud_bytes = aud_array.tobytes()

        # Construct the payload: header length + header bytes + audio bytes
        payload = header_length_packed + header_bytes + aud_bytes

        # Publish the binary payload
        try:
            MqttMiddlewarePubSub._instance.mqtt_client.publish(self.out_topic, payload)
        except Exception as e:
            logging.error(f"Failed to publish message: {e}")


@Publishers.register("Properties", "mqtt")
class MqttPropertiesPublisher(MqttPublisher):

    def __init__(self, name, out_topic, **kwargs):
        super().__init__(name, out_topic, **kwargs)
        raise NotImplementedError
