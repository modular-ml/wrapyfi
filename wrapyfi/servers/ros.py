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
from wrapyfi.middlewares.ros import ROSMiddleware, ROSNativeObjectService
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


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
    SEND_QUEUE = queue.Queue(maxsize=1)
    RECEIVE_QUEUE = queue.Queue(maxsize=1)

    def __init__(self, name, out_port, carrier="", out_port_connect=None, serializer_kwargs=None, deserializer_kwargs=None, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        self._server = None

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

    def establish(self):
        self._server = rospy.Service(self.out_port, ROSNativeObjectService, self._service_callback)
        self.established = True

    def await_request(self, *args, **kwargs):
        if not self.established:
            self.establish()
        try:
            request = ROSNativeObjectServer.RECEIVE_QUEUE.get(block=True)
            [args, kwargs] = json.loads(request.data, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
            return args, kwargs
        except rospy.ServiceException as e:
            logging.error("Service call failed: %s" % e)
            return [], {}

    @staticmethod
    def _service_callback(msg):
       ROSNativeObjectServer.RECEIVE_QUEUE.put(msg)
       return ROSNativeObjectServer.SEND_QUEUE.get(block=True)

    def reply(self, obj):
        try:
            obj_str = json.dumps(obj, cls=self._plugin_encoder, **self._plugin_kwargs,
                                 serializer_kwrags=self._serializer_kwargs)
            obj_msg = std_msgs.msg.String()
            obj_msg.data = obj_str
            ROSNativeObjectServer.SEND_QUEUE.put(obj_msg)
        except queue.Full:
            logging.warning(f"Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
