import logging
import sys
import json
import time
import os
import importlib.util
import queue

import numpy as np
import yarp

from wrapyfi.connect.clients import Client, Clients
from wrapyfi.middlewares.yarp import YarpMiddleware
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class YarpClient(Client):

    def __init__(self, name, in_port, carrier="", ros_kwargs=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        YarpMiddleware.activate(**ros_kwargs or {})

    def close(self):
        if hasattr(self, "_port") and self._port:
            if self._port is not None:
                self._port.close()

    def __del__(self):
        self.close()


@Clients.register("NativeObject", "yarp")
class YarpNativeObjectClient(YarpClient):

    def __init__(self, name, in_port, carrier="", serializer_kwargs=None, deserializer_kwargs=None, persistent=False, **kwargs):
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        self._port = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self.persistent = persistent

    def establish(self):
        while not yarp.Network.exists(self.in_port):
            logging.info(f"Waiting for input port: {self.in_port}")
            time.sleep(0.2)
        self._port = yarp.RpcClient()
        rnd_id = str(np.random.randint(100000, size=1)[0])
        self._port.open(self.in_port + ":in" + rnd_id)
        self._port.addOutput(self.in_port, self.carrier)
        if self.persistent:
            self.established = True

    def request(self, *args, **kwargs):
        if not self.established:
            self.establish()
        try:
            self._request(*args, **kwargs)
        except Exception as e:
            logging.error("Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        # transmit args to server
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = yarp.Bottle()
        args_msg.clear()
        args_msg.addString(args_str)
        # receive message from server
        msg = yarp.Bottle()
        msg.clear()
        self._port.write(args_msg, msg)
        obj = json.loads(msg.get(0).asString(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        self._queue.put(obj, block=False)

    def _await_reply(self):
        try:
            reply = self._queue.get(block=True)
            return reply
        except queue.Full:
            logging.warning(f"Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
            return None



@Clients.register("Image", "yarp")
class YarpImageClient(YarpNativeObjectClient):

    def __init__(self, name, in_port, carrier="",  width=-1, height=-1, rgb=True, fp=False, **kwargs):
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp

    def _request(self, *args, **kwargs):
        # transmit args to server
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                              serializer_kwrags=self._serializer_kwargs)
        args_msg = yarp.Bottle()
        args_msg.clear()
        args_msg.addString(args_str)
        # receive message from server
        # receive message from server
        msg = yarp.Bottle()
        msg.clear()
        self._port.write(args_msg, msg)
        img = json.loads(msg.get(0).asString(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0]:
            raise ValueError("Incorrect image shape for listener")
        else:
            self._queue.put(img, block=False)
