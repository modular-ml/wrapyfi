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


from wrapyfi.connect.clients import Client, Clients
from wrapyfi.middlewares.ros import ROSMiddleware, ROSNativeObjectService
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class ROSClient(Client):

    def __init__(self, name, in_port, carrier="", ros_kwargs=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        ROSMiddleware.activate(**ros_kwargs or {})

    def close(self):
        if hasattr(self, "_client"):
            self._client.shutdown()

    def __del__(self):
        self.close()


@Clients.register("NativeObject", "ros")
class ROSNativeObjectClient(ROSClient):

    def __init__(self, name, in_port, carrier="", serializer_kwargs=None, deserializer_kwargs=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        self._client = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._reply_func = None

    def request(self, *args, **kwargs):
        rospy.wait_for_service(self.in_port)
        try:
            self._client = rospy.ServiceProxy(self.in_port, ROSNativeObjectService)
            self._request(*args, **kwargs)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        return self._await_reply()

    def _request(self, *args, **kwargs):
        args_str = json.dumps([args, kwargs], cls=self._plugin_encoder, **self._plugin_kwargs,
                             serializer_kwrags=self._serializer_kwargs)
        args_msg = std_msgs.msg.String()
        args_msg.data = args_str
        msg = self._client(args_msg)
        obj = json.loads(msg.data, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
        self._queue.put(obj)

    def _await_reply(self):
        try:
            reply = self._queue.get(block=True)
            return reply
        except queue.Full:
            logging.warning(f"Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
