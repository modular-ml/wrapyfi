import logging
import os
import json
from typing import Optional, Tuple

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
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for pub/sub pattern. Default is 'tcp'
        :param socket_ip: str: IP address of the socket. Default is '127.0.0.1'
        :param socket_rep_port: int: Port of the socket for REP pattern. Default is 5558
        :param socket_req_port: int: Port of the socket for REQ pattern. Default is 5559
        :param start_proxy_broker: bool: Whether to start a device broker. Default is True
        :param proxy_broker_spawn: str: Whether to spawn the device broker as a process or thread. Default is 'process'
        :param zeromq_kwargs: dict: Additional kwargs for the ZeroMQ Req/Rep middleware
        :param kwargs: Additional kwargs for the server
        """
        if carrier != "tcp":
            logging.warning("[ZeroMQ] ZeroMQ does not support other carriers than TCP for pub/sub pattern. Using TCP.")
            carrier = "tcp"
        super().__init__(name, out_topic, carrier=carrier, **kwargs)

        # out_topic is equivalent to topic in zeromq
        self.socket_rep_address = f"{carrier}://{socket_ip}:{socket_rep_port}"
        self.socket_req_address = f"{carrier}://{socket_ip}:{socket_req_port}"
        if start_proxy_broker:
            ZeroMQMiddlewareReqRep.activate(socket_rep_address=self.socket_rep_address,
                                            socket_req_address=self.socket_req_address,
                                            proxy_broker_spawn=proxy_broker_spawn,
                                            **zeromq_kwargs or {})
        else:
            ZeroMQMiddlewareReqRep.activate(**zeromq_kwargs or {})

        self._socket = zmq.Context().instance().socket(zmq.REP)
        self._socket.connect(self.socket_req_address)

    def await_request(self, *args, **kwargs):
        """
        Wait for the request from the client and return it
        """
        return self._socket.recv_string()

    def reply(self, message):
        """
        Send reply back to the client

        :param message: str: Message to be sent to the client
        """
        self._socket.send_string(message)

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
        :param out_topic: str: Name of the output topic preceded by '/' (e.g. '/topic')
        :param carrier: str: Carrier protocol. ZeroMQ currently only supports TCP for pub/sub pattern. Default is 'tcp'
        :param serializer_kwargs: dict: Additional kwargs for the serializer
        :param deserializer_kwargs: dict: Additional kwargs for the deserializer
        :param kwargs: Additional kwargs for the ZeroMQServer
        """
        super().__init__(name, out_topic, carrier=carrier, **kwargs)
        self._plugin_encoder = JsonEncoder
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

    # def establish(self, repeats: Optional[int] = None, **kwargs):
    #     """
    #     Establish the connection to the server
    #
    #     :param repeats: int: Number of repeats to await connection. None for infinite. Default is None
    #     :return: bool: True if connection established, False otherwise
    #     """
    #     self._socket = zmq.Context.instance().socket(zmq.REP)
    #     for socket_property in ZeroMQMiddlewareReqRep().zeromq_kwargs.items():
    #         if isinstance(socket_property[1], str):
    #             self._socket.setsockopt_string(getattr(zmq, socket_property[0]), socket_property[1])
    #         else:
    #             self._socket.setsockopt(getattr(zmq, socket_property[0]), socket_property[1])
    #     self._socket.connect(self.socket_req_address)
    #     self._topic = self.out_topic.encode()
    #     established = self.await_connection(self._socket, repeats=repeats)
    #     return self.check_establishment(established)

    def await_request(self, *args, **kwargs):
        message = super().await_request(*args, **kwargs)
        try:
            request = json.loads(message, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
            args, kwargs = request
            return args, kwargs
        except json.JSONDecodeError as e:
            logging.error(f"[ZeroMQ] Failed to decode message: {e}")
            return None, None

    def reply(self, obj):
        obj_str = json.dumps(obj, cls=self._plugin_encoder, **self._serializer_kwargs)
        super().reply(obj_str)