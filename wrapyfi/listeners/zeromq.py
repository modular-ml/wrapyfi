import logging
import json
import time
import os

import numpy as np
import zmq

from wrapyfi.connect.listeners import Listener, ListenerWatchDog, Listeners
from wrapyfi.middlewares.zeromq import ZeroMQMiddlewarePubSub
from wrapyfi.encoders import JsonDecodeHook


SOCKET_IP = os.environ.get("WRAPYFI_ZEROMQ_SOCKET_IP", "127.0.0.1")
SOCKET_PORT = int(os.environ.get("WRAPYFI_ZEROMQ_SOCKET_PORT", 5555))
WATCHDOG_POLL_REPEAT = None


class ZeroMQListener(Listener):

    def __init__(self, name, in_port, carrier="tcp",
                 socket_ip=SOCKET_IP, socket_port=SOCKET_PORT, zeromq_kwargs=None, **kwargs):
        carrier = carrier if carrier else "tcp"
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        ZeroMQMiddlewarePubSub.activate(**zeromq_kwargs or {})
        self.socket_address = f"{carrier}://{socket_ip}:{socket_port}"

    def await_connection(self, socket=None, port_in=None, repeats=None):
        connected = False
        if port_in is None:
            port_in = self.in_port
        logging.info(f"Waiting for input port: {port_in}")
        if repeats is None:
            if self.should_wait:
                repeats = -1
            else:
                repeats = 1

            while repeats > 0 or repeats <= -1:
                repeats -= 1
                # TODO (fabawi): communicate with proxy broker to check whether publisher exists
                connected = True
                if connected:
                    logging.info(f"Connected to input port: {port_in}")
                    break
                time.sleep(0.2)
        return connected

    def read_port(self, socket):
        while True:
            obj = socket.read(shouldWait=False)
            if self.should_wait and obj is None:
                time.sleep(0.005)
            else:
                return obj

    def close(self):
        if hasattr(self, "_socket") and self._socket:
            if self._socket is not None:
                self._socket.close()

    def __del__(self):
        self.close()


@Listeners.register("NativeObject", "zeromq")
class ZeroMQNativeObjectListener(ZeroMQListener):

    def __init__(self, name, in_port, carrier="tcp", deserializer_kwargs=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        self._socket = self._netconnect = None

        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        if not self.should_wait:
            ListenerWatchDog().add_listener(self)

    def establish(self, repeats=None, **kwargs):
        established = self.await_connection(repeats=repeats)
        if established:
            self._socket = zmq.Context.instance().socket(zmq.SUB)
            for socket_property in ZeroMQMiddlewarePubSub().zeromq_kwargs.items():
                if isinstance(socket_property[1], str):
                    self._socket.setsockopt(*socket_property)
                else:
                    self._socket.setsockopt(*socket_property)
            self._socket.connect(self.socket_address)
            self._topic = self.in_port.encode("utf-8")
            self._socket.setsockopt_string(zmq.SUBSCRIBE, self.in_port)

        return self.check_establishment(established)

    def listen(self):
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return None
        if self._socket.poll(timeout=None if self.should_wait else 0):
            obj = self._socket.recv_multipart()
            if obj is not None:
                return json.loads(obj[1].decode(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
            else:
                return None
        else:
            return None


@Listeners.register("Image", "zeromq")
class ZeroMQImageListener(ZeroMQNativeObjectListener):

    def __init__(self, name, out_port, carrier="tcp", out_port_connect=None, width=-1, height=-1, rgb=True, fp=False,
                 **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        self._type = np.float32 if self.fp else np.uint8

    def listen(self):
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return None
        if self._socket.poll(timeout=None if self.should_wait else 0):
            obj = self._socket.recv_multipart()
            img = json.loads(obj[1].decode(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs) if obj is not None else None
            if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                    not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
                raise ValueError("Incorrect image shape for listener")
            return img
        else:
            return None


@Listeners.register("AudioChunk", "zeromq")
class ZeroMQAudioChunkListener(ZeroMQImageListener):
    def __init__(self, name, in_port, carrier="tcp", channels=1, rate=44100, chunk=-1, **kwargs):
        super().__init__(name, in_port, carrier=carrier, width=chunk, height=channels, rgb=False, fp=True, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def listen(self):
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return None
        if self._socket.poll(timeout=None if self.should_wait else 0):
            obj = self._socket.recv_multipart()
            aud = json.loads(obj[1].decode(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs) if obj is not None else None

            chunk, channels = aud.shape[0], aud.shape[1]
            if 0 < self.chunk != chunk or self.channels != channels or len(aud) != chunk * channels * 4:
                raise ValueError("Incorrect audio shape for listener")
            return aud, self.rate
        else:
            return None, self.rate


@Listeners.register("Properties", "zeromq")
class ZeroMQPropertiesListener(ZeroMQListener):
    def __init__(self, name, in_port, **kwargs):
        super().__init__(name, in_port, **kwargs)
        raise NotImplementedError
