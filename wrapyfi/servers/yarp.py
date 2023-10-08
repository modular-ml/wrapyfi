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

    def __init__(self, name, out_topic, carrier="", out_topic_connect=None, yarp_kwargs=None, **kwargs):
        super().__init__(name, out_topic, carrier=carrier, out_topic_connect=out_topic_connect, **kwargs)
        YarpMiddleware.activate(**yarp_kwargs or {})

    def close(self):
        if hasattr(self, "_port") and self._port:
            if self._port is not None:
                self._port.close()

    def __del__(self):
        self.close()


@Servers.register("NativeObject", "yarp")
class YarpNativeObjectServer(YarpServer):

    def __init__(self, name, out_topic, carrier="", out_topic_connect=None, serializer_kwargs=None, deserializer_kwargs=None, persistent=False, **kwargs):
        super().__init__(name, out_topic, carrier=carrier, out_topic_connect=out_topic_connect, **kwargs)

        self._port = self._netconnect = None

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self.persistent = persistent

    def establish(self):
        self._port = yarp.RpcServer()
        self._port.open(self.out_topic)
        self._netconnect = yarp.Network.connect(self.out_topic, self.out_topic_connect, self.carrier)
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
            logging.error("[YARP] Service call failed: %s" % e)
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


@Servers.register("Image", "yarp")
class YarpImageServer(YarpNativeObjectServer):

    def __init__(self, name, out_topic, carrier="", out_topic_connect=None, width=-1, height=-1, rgb=True, fp=False, **kwargs):
        super().__init__(name, out_topic, carrier=carrier, out_topic_connect=out_topic_connect, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp

    def reply(self, img):
        if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
            raise ValueError("Incorrect image shape for publisher")
        # img = np.require(img, dtype=self._type, requirements='C')
        super().reply(img)

@Servers.register("Image", "yarp")
class YarpImageServer(YarpNativeObjectServer):

    def __init__(self, name, out_topic, carrier="", out_topic_connect=None, width=-1, height=-1, rgb=True, fp=False, **kwargs):
        super().__init__(name, out_topic, carrier=carrier, out_topic_connect=out_topic_connect, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp

    def reply(self, img):
        if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
            raise ValueError("Incorrect image shape for publisher")
        # img = np.require(img, dtype=self._type, requirements='C')
        super().reply(img)