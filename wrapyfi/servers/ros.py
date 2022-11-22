import logging
import sys
import json
import time
import os
import importlib.util
import queue

import numpy as np
import rospy
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.servers import Server, Servers
from wrapyfi.middlewares.ros import ROSMiddleware
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class ROSService(object):
  _type          = 'wrapyfi_services/ROSService'
  # _md5sum = '6a2e34150c00229791cc89ff309fff21'
  _request_class  = std_msgs.msg.String
  _response_class = std_msgs.msg.String


class ROSServer(Server):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, ros_kwargs=None, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        ROSMiddleware.activate(**ros_kwargs or {})

    def close(self):
        if hasattr(self, "_server"):
            self._server.shutdown()

    def __del__(self):
        self.close()


@Servers.register("NativeObject", "ros")
class ROSNativeObjectServer(ROSServer):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, serializer_kwargs=None, deserializer_kwargs=None, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        self._server = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._reply_func = None

    def await_request(self, func, **kwargs):
        self._reply_func = func
        self._server = rospy.Service(self.out_port, ROSService, self._reply)
        return self._await_reply()

    def _reply(self, msg):
        args, kwargs = json.loads(msg.data, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        obj = self._reply_func(*args, **kwargs)
        self._queue.put(obj)
        obj_str = json.dumps(obj, cls=self._plugin_encoder, **self._plugin_kwargs,
                             serializer_kwrags=self._serializer_kwargs)
        return obj_str

    def _await_reply(self):
        try:
            reply = self._queue.get(block=True)
            return reply
        except queue.Full:
            logging.warning(f"Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
