import logging
import sys
import json
import time
import os
import importlib.util
import queue

import numpy as np
import yarp

from wrapyfi.connect.servers import Server, Servers
from wrapyfi.middlewares.yarp import YarpMiddleware
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class YarpServer(Server):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, yarp_kwargs=None, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        YarpMiddleware.activate(**yarp_kwargs or {})

    def close(self):
        if hasattr(self, "_port") and self._port:
            if self._port is not None:
                self._port.close()

    def __del__(self):
        self.close()


@Servers.register("NativeObject", "yarp")
class YarpNativeObjectServer(YarpServer):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, serializer_kwargs=None, deserializer_kwargs=None, persistent=False, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        self._port = self._netconnect = None

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self.persistent = persistent

    def establish(self):
        self._port = yarp.RpcServer()
        self._port.open(self.out_port)
        self._netconnect = yarp.Network.connect(self.out_port, self.out_port_connect, self.carrier)
        if self.persistent:
            self.established = True

    def await_request(self, *args, **kwargs):
        if not self.established:
            self.establish()
        try:
            obj_msg = yarp.Bottle()
            obj_msg.clear()
            request = False
            while not request:
                request = self._port.read(obj_msg, True)
            [args, kwargs] = json.loads(obj_msg.get(0).asString(), object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
            return args, kwargs
        except Exception as e:
            logging.error("Service call failed: %s" % e)
            return [], {}

    def reply(self, obj):
        obj_str = json.dumps(obj, cls=self._plugin_encoder, **self._plugin_kwargs,
                             serializer_kwrags=self._serializer_kwargs)
        obj_msg = yarp.Bottle()
        obj_msg.clear()
        obj_msg.addString(obj_str)
        if self.persistent:
            self._port.reply(obj_msg)
        else:
            self._port.replyAndDrop(obj_msg)

