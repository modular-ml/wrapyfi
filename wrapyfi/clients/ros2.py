import logging
import sys
import json
import time
import os
import importlib.util
import queue

import numpy as np
import rclpy
from rclpy.node import Node
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.clients import Client, Clients
from wrapyfi.middlewares.ros2 import ROS2Middleware
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class ROS2Client(Client, Node):

    def __init__(self, name, in_port, carrier="", ros2_kwargs=None, **kwargs):
        ROS2Middleware.activate(**ros2_kwargs or {})
        Client.__init__(name, in_port, carrier=carrier, **kwargs)
        Node.__init__(self, name)

    def close(self):
        if hasattr(self, "_client"):
            if self._client is not None:
                self.destroy_node()

    def __del__(self):
        self.close()


@Clients.register("NativeObject", "ros2")
class ROS2NativeObjectClient(ROS2Client):

    def __init__(self, name, in_port, carrier="", serializer_kwargs=None, deserializer_kwargs=None, **kwargs):
        super().__init__(name, in_port, carrier=carrier, **kwargs)
        self._client = None
        self._queue = queue.Queue(maxsize=1)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

    def establish(self):
        # TODO (fabawi): add documentation on compiling the service
        from wrapyfi_interfaces.ros2.srv import ROS2NativeObjectService
        self._client = self.create_client(ROS2NativeObjectService, self.in_port)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            logging.info('Service not available, waiting again...')
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
        args_msg = std_msgs.msg.String()
        args_msg.data = args_str
        future = self._client.call_async(args_msg)
        # receive message from server
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    msg = future.result()
                    obj = json.loads(msg.data, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
                    self._queue.put(obj, block=False)
                except Exception as e:
                    logging.error("Service call failed: %s" % e)
                break

    def _await_reply(self):
        try:
            reply = self._queue.get(block=True)
            return reply
        except queue.Full:
            logging.warning(f"Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")
            return None
