import logging
import time
import os
import queue
import json
from typing import Optional, Tuple
import random
import struct

import numpy as np
import cv2

from wrapyfi.connect.listeners import Listener, Listeners, ListenerWatchDog
from wrapyfi.encoders import JsonDecodeHook
from wrapyfi.middlewares.mqtt import MqttMiddlewarePubSub

MQTT_BROKER_ADDRESS = os.environ.get("WRAPYFI_MQTT_BROKER_ADDRESS", "broker.emqx.io")
MQTT_BROKER_PORT = int(os.environ.get("WRAPYFI_MQTT_BROKER_PORT", 1883))
WATCHDOG_POLL_REPEAT = None


class MqttListener(Listener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        broker_address: str = MQTT_BROKER_ADDRESS,
        broker_port: int = MQTT_BROKER_PORT,
        mqtt_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the subscriber.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param broker_address: str: Address of the MQTT broker. Default is 'broker.emqx.io'
        :param broker_port: int: Port of the MQTT broker. Default is 1883
        :param mqtt_kwargs: dict: Additional kwargs for the MQTT middleware
        :param kwargs: dict: Additional kwargs for the subscriber
        """
        super().__init__(name, in_topic, should_wait=should_wait, **kwargs)

        if mqtt_kwargs is None:
            mqtt_kwargs = {}

        if "client_id" not in mqtt_kwargs:
            mqtt_kwargs["client_id"] = f"client-{name}-{random.randint(0, 100000)}"
        mqtt_kwargs["clean_session"] = False

        # Activate the MQTT middleware
        MqttMiddlewarePubSub.activate(
            broker_address=broker_address, broker_port=broker_port, **(mqtt_kwargs or {})
        )

        if not self.should_wait:
            ListenerWatchDog().add_listener(self)

    def await_connection(self, in_topic: Optional[str] = None, repeats: Optional[int] = None):
        """
        Wait until the MQTT connection is established.

        :param in_topic: str: The topic to monitor for connection
        :param repeats: int: The number of times to check for the connection, None for infinite.
        """
        if in_topic is None:
            in_topic = self.in_topic
        logging.info(f"[MQTT] Waiting for input topic: {in_topic}")
        if repeats is None:
            repeats = -1 if self.should_wait else 0

        # Ensure to call is_connected() on the singleton instance
        while repeats > 0 or repeats == -1:
            if repeats != -1:
                repeats -= 1
            connected = (
                MqttMiddlewarePubSub._instance.is_connected()
            )  # Use the instance
            logging.debug(f"Connection status: {connected}")
            if connected:
                logging.info(f"[MQTT] Connected to input topic: {in_topic}")
                return True
            time.sleep(0.2)
        return False

    def close(self):
        """
        Close the subscriber.
        """
        pass

    def __del__(self):
        self.close()


@Listeners.register("NativeObject", "mqtt")
class MqttNativeObjectListener(MqttListener):

    def __init__(
        self,
        name: str,
        in_topic: str,
        should_wait: bool = True,
        deserializer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        The NativeObject listener using the MQTT message construct assuming the data is serialized as a JSON string.
        Deserializes the data (including plugins) using the decoder and parses it to a native object.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic
        :param should_wait: bool: Whether the subscriber should wait for the publisher to transmit a message. Default is True
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        """
        super().__init__(name, in_topic, should_wait=should_wait, **kwargs)
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}
        self._message_queue = queue.Queue()

    def on_message(self, client, userdata, msg):
        """
        Callback for handling incoming MQTT messages.
        """
        try:
            obj = json.loads(
                msg.payload.decode(),
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
        established = self.check_establishment(established)
        if established:
            MqttMiddlewarePubSub._instance.register_callback(self.in_topic, self.on_message)
        return established

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


@Listeners.register("Image", "mqtt")
class MqttImageListener(MqttNativeObjectListener):

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
        The Image listener using the MQTT message construct parsed to a numpy array.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic
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

    def on_message(self, client, userdata, msg):
        """
        Callback for handling incoming image messages.
        """
        try:
            payload = msg.payload
            # Read the first 4 bytes to get the header length
            header_length_packed = payload[:4]
            header_length = struct.unpack('!I', header_length_packed)[0]
            # Read the header bytes
            header_bytes = payload[4:4 + header_length]
            header_json = header_bytes.decode('utf-8')
            header = json.loads(header_json)
            # Remaining bytes are image bytes
            img_bytes = payload[4 + header_length:]

            if self.jpg:
                # JPEG case: decode the JPEG image
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                if self.rgb:
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                else:
                    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                self._message_queue.put(img)
            else:
                # Non-JPEG case: reconstruct the numpy array
                shape = header.get("shape", None)
                dtype = header.get("dtype", None)
                if shape is not None and dtype is not None:
                    img_array = np.frombuffer(img_bytes, dtype=dtype)
                    img = img_array.reshape(shape)
                    self._message_queue.put(img)
                else:
                    logging.error("Missing 'shape' or 'dtype' in header for non-JPEG image")
        except Exception as e:
            logging.error(f"Failed to process message from topic {self.in_topic}: {e}")

    def listen(self) -> Optional[np.ndarray]:
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
                    (self.width > 0 and self.width != img.shape[1])
                    or (self.height > 0 and self.height != img.shape[0])
                    or not (
                    (img.ndim == 2 and not self.rgb)
                    or (img.ndim == 3 and self.rgb and img.shape[2] == 3)
            )
            ):
                raise ValueError("Incorrect image shape for listener")
            return img
        except queue.Empty:
            return None


@Listeners.register("AudioChunk", "mqtt")
class MqttAudioChunkListener(MqttNativeObjectListener):

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
        The AudioChunk listener using the MQTT message construct parsed to a numpy array.

        :param name: str: Name of the subscriber
        :param in_topic: str: Name of the input topic
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

    def on_message(self, client, userdata, msg):
        """
        Callback for handling incoming audio chunk messages.
        """
        try:
            payload = msg.payload
            # Ensure payload is long enough to contain the header length
            if len(payload) < 4:
                logging.error("Payload too short to contain header length")
                return

            # Read the first 4 bytes to get the header length
            header_length_packed = payload[:4]
            header_length = struct.unpack('!I', header_length_packed)[0]

            # Ensure payload is long enough to contain the header
            if len(payload) < 4 + header_length:
                logging.error("Payload too short to contain header")
                return

            # Read the header bytes
            header_bytes = payload[4:4 + header_length]
            header_json = header_bytes.decode('utf-8')
            header = json.loads(header_json)

            # Remaining bytes are audio bytes
            aud_bytes = payload[4 + header_length:]

            # Extract metadata from header
            shape = header.get("shape")
            dtype = header.get("dtype")
            rate = header.get("rate")
            timestamp = header.get("timestamp")

            if shape is None or dtype is None or rate is None:
                logging.error("Missing 'shape', 'dtype', or 'rate' in header")
                return

            if 0 < self.rate != rate:
                raise ValueError("Incorrect audio rate for listener")

            # Reconstruct the audio array from the binary data
            aud_array = np.frombuffer(aud_bytes, dtype=dtype)
            if aud_array.size != np.prod(shape):
                logging.error("Mismatch between audio data size and shape")
                return

            aud_array = aud_array.reshape(shape)

            chunk, channels = (
                aud_array.shape if len(aud_array.shape) > 1 else (aud_array.shape[0], 1)
            )

            if (
                    (0 < self.chunk != chunk)
                    or (0 < self.channels != channels)
            ):
                raise ValueError("Incorrect audio shape for listener")

            self._message_queue.put((aud_array, rate))
        except Exception as e:
            logging.error(f"Failed to process message from topic {self.in_topic}: {e}")

    def establish(self, repeats: Optional[int] = None, **kwargs):
        """
        Establish the connection to the publisher.

        :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
        :return: bool: True if connection established, False otherwise
        """
        return super().establish(repeats=repeats, **kwargs)

    def listen(self) -> Tuple[Optional[np.ndarray], int]:
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


@Listeners.register("Properties", "mqtt")
class MqttPropertiesListener(MqttNativeObjectListener):
    def __init__(self, name, in_topic, **kwargs):
        super().__init__(name, in_topic, **kwargs)
        raise NotImplementedError
