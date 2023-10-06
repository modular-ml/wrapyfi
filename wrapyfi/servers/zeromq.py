import logging
import json
import time
import os
from typing import Optional, Tuple

import numpy as np
import cv2
import zmq

from wrapyfi.connect.servers import Server, Servers
from wrapyfi.middlewares.zeromq import ZeroMQMiddlewareReqRep
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


SOCKET_IP = os.environ.get("WRAPYFI_ZEROMQ_SOCKET_IP", "127.0.0.1")
SOCKET_PUB_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_SOCKET_REQ_PORT", 5558))
SOCKET_SUB_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_SOCKET_REP_PORT", 5559))
START_PROXY_BROKER = os.environ.get("WRAPYFI_ZEROMQ_START_PROXY_BROKER", True) != "False"
PROXY_BROKER_SPAWN = os.environ.get("WRAPYFI_ZEROMQ_PROXY_BROKER_SPAWN", "process")
WATCHDOG_POLL_REPEAT = None


class ZeroMQServer(Server):
    def __init__(self, name: str, out_topic: str, carrier: str = "tcp",
                 socket_ip: str = SOCKET_IP, socket_rep_port: int = SOCKET_PUB_PORT, socket_req_port: int = SOCKET_SUB_PORT,
                 start_proxy_broker: bool = START_PROXY_BROKER, proxy_broker_spawn: bool = PROXY_BROKER_SPAWN,
                 zeromq_kwargs: Optional[dict] = None, **kwargs):
        """
        Initialize the server and start the device broker if necessary

        :param name: str: Name of the server
        :param out_topic: str: Topics are not supported for the REQ/REP pattern in ZeroMQ. Any given topic is ignored
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param socket_ip: str: IP address of the socket. Default is '127.0.0.1'
        :param socket_rep_port: int: Port of the socket for REP pattern. Default is 5558
        :param socket_req_port: int: Port of the socket for REQ pattern. Default is 5559
        :param start_proxy_broker: bool: Whether to start a device broker. Default is True
        :param proxy_broker_spawn: str: Whether to spawn the device broker as a process or thread. Default is 'process'
        :param zeromq_kwargs: dict: Additional kwargs for the ZeroMQ Req/Rep middleware
        :param kwargs: Additional kwargs for the server
        """
        if out_topic != "":
            logging.warning(f"[ZeroMQ] ZeroMQ does not support topics for the REQ/REP pattern. Topic {out_topic} removed")
            out_topic = ""
        if carrier or carrier != "tcp":
            logging.warning("[ZeroMQ] ZeroMQ does not support other carriers than TCP for REQ/REP pattern. Using TCP.")
            carrier = "tcp"
        super().__init__(name, out_topic, carrier=carrier, **kwargs)

        self.socket_rep_address = f"{carrier}://{socket_ip}:{socket_rep_port}"
        self.socket_req_address = f"{carrier}://{socket_ip}:{socket_req_port}"
        if start_proxy_broker:
            ZeroMQMiddlewareReqRep.activate(socket_rep_address=self.socket_rep_address,
                                            socket_req_address=self.socket_req_address,
                                            proxy_broker_spawn=proxy_broker_spawn,
                                            **zeromq_kwargs or {})
        else:
            ZeroMQMiddlewareReqRep.activate(**zeromq_kwargs or {})

    def close(self):
        """
        Close the publisher
        """
        if hasattr(self, "_socket") and self._socket:
            if self._socket is not None:
                self._socket.close()

    def __del__(self):
        self.close()


@Servers.register("NativeObject", "zeromq")
class ZeroMQNativeObjectServer(ZeroMQServer):
    def __init__(self, name: str, out_topic: str, carrier: str = "tcp",
                 serializer_kwargs: Optional[dict] = None, deserializer_kwargs: Optional[dict] = None, **kwargs):
        """
        Specific server handling native Python objects, serializing them to JSON strings for transmission.

        :param name: str: Name of the server
        :param out_topic: str: Topics are not supported for the REQ/REP pattern in ZeroMQ. Any given topic is ignored
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for PUB/SUB pattern. Default is 'tcp'
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        :param kwargs: Additional kwargs for the server
        """
        super().__init__(name, out_topic, carrier=carrier, **kwargs)
        self._plugin_encoder = JsonEncoder
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

    def establish(self, **kwargs):
        """
        Establish the connection to the publisher
        """
        self._socket = zmq.Context().instance().socket(zmq.REP)
        for socket_property in ZeroMQMiddlewareReqRep().zeromq_kwargs.items():
            if isinstance(socket_property[1], str):
                self._socket.setsockopt_string(getattr(zmq, socket_property[0]), socket_property[1])
            else:
                self._socket.setsockopt(getattr(zmq, socket_property[0]), socket_property[1])
        self._socket.connect(self.socket_req_address)
        self.established = True

    def await_request(self, *args, **kwargs):
        """
        Await and deserialize the client's request, returning the extracted arguments and keyword arguments.
        The method blocks until a message is received, then attempts to deserialize it using the configured JSON decoder hook, returning the extracted arguments and keyword arguments. If the message cannot be deserialized, it logs an error and returns empty argument and keyword argument sets.

        :return: Tuple[list, dict]: A tuple containing two items:
                 - A list of arguments extracted from the received message
                 - A dictionary of keyword arguments extracted from the received message
        """
        if not self.established:
            self.establish()
        message = self._socket.recv_string()
        try:
            request = json.loads(message, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
            args, kwargs = request
            return args, kwargs
        except json.JSONDecodeError as e:
            logging.error(f"[ZeroMQ] Failed to decode message: {e}")
            return [], {}

    def reply(self, obj):
        """
        Serialize the provided Python object to a JSON string and send it as a reply to the client.
        The method uses the configured JSON encoder for serialization before sending the resultant string to the client.

        :param obj: Any: The Python object to be serialized and sent
        """
        obj_str = json.dumps(obj, cls=self._plugin_encoder, **self._serializer_kwargs)
        self._socket.send_string(obj_str)


@Servers.register("Image", "zeromq")
class ZeroMQImageServer(ZeroMQNativeObjectServer):

    def __init__(self, name: str, out_topic: str, carrier: str = "tcp",
                 serializer_kwargs: Optional[dict] = None, deserializer_kwargs: Optional[dict] = None,
                 width: int = -1, height: int = -1, rgb: bool = True, fp: bool = False, jpg: bool = False, **kwargs):
        """
        Specific server handling image data


        """
        super().__init__(name, out_topic, carrier=carrier,
                         serializer_kwargs=serializer_kwargs, deserializer_kwargs=deserializer_kwargs, **kwargs)

        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self.jpg = jpg

        self._type = np.float32 if self.fp else np.uint8

    def reply(self, img: np.ndarray):
        """
        Serialize the provided image data and send it as a reply to the client.

        :param img: np.ndarray: Image to send formatted as a cv2 image np.ndarray[img_height, img_width, channels]
        """
        if img is None:
            logging.warning("[ZeroMQ] Image is None. Skipping reply.")
            return

        if not self.established:
            self.establish()
            time.sleep(0.2)  # To ensure the connection setup is complete before sending

        if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
            raise ValueError("Incorrect image shape for server reply")

        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)

        if self.jpg:
            _, img_encoded = cv2.imencode('.jpg', img)
            img_bytes = img_encoded.tobytes()
            self._socket.send(img_bytes)
        else:
            img_list = img.tolist()
            img_json = json.dumps({"img": img_list})
            self._socket.send_string(img_json)
