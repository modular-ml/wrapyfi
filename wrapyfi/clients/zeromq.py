import logging
import json
import queue
import os
from typing import Optional

import zmq

from wrapyfi.connect.clients import Client, Clients
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook

SOCKET_IP = os.environ.get("WRAPYFI_ZEROMQ_SOCKET_IP", "127.0.0.1")
SOCKET_REP_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_SOCKET_REP_PORT", 5558))
WATCHDOG_POLL_REPEAT = None


class ZeroMQClient(Client):
    def __init__(self, name, in_topic, carrier="tcp",
                 socket_ip: str = SOCKET_IP, socket_rep_port: int = SOCKET_REP_PORT, zeromq_kwargs: Optional[dict] = None, **kwargs):
        if carrier != "tcp":
            logging.warning("ZeroMQ does not support other carriers than TCP for pub/sub pattern. Using TCP.")
            carrier = "tcp"
        super().__init__(name, in_topic, carrier=carrier, **kwargs)

        self.socket_address = f"{carrier}://{socket_ip}:{socket_rep_port}"

        self._socket = zmq.Context().instance().socket(zmq.REQ)
        self._socket.connect(self.socket_address)

    def close(self):
        """
        Close the subscriber
        """
        if hasattr(self, "_socket") and self._socket:
            if self._socket is not None:
                self._socket.close()

    def __del__(self):
        self.close()


@Clients.register("NativeObject", "zeromq")
class ZeroMQNativeObjectClient(ZeroMQClient):
    def __init__(self, name, in_topic, carrier="tcp", serializer_kwargs=None, deserializer_kwargs=None, **kwargs):
        super().__init__(name, in_topic, carrier=carrier, **kwargs)

        self._plugin_encoder = JsonEncoder
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._queue = queue.Queue(maxsize=1)

    def request(self, *args, **kwargs):
        try:
            self._request(*args, **kwargs)
        except zmq.ZMQError as e:
            logging.error("Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._serializer_kwargs)
        self._socket.send_string(args_str)

        # await reply from server
        reply_str = self._socket.recv_string()
        reply = json.loads(reply_str, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        self._queue.put(reply, block=False)

    def _await_reply(self):
        try:
            reply = self._queue.get(block=True)
            return reply
        except queue.Empty:
            logging.warning(f"Discarding data because queue is empty. "
                            f"This happened due to bad synchronization in {self.__name__}")
            return None