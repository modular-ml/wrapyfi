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

from wrapyfi.connect.servers import Server, Servers
from wrapyfi.middlewares.ros2 import ROS2Middleware
from wrapyfi.encoders import JsonEncoder, JsonDecodeHook


class ROS2Server(Server, Node):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, ros2_kwargs=None, **kwargs):
        ROS2Middleware.activate(**ros2_kwargs or {})
        Server.__init__(self, name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        Node.__init__(self, name)

    def close(self):
        if hasattr(self, "_server"):
            self.destroy_node()

    def __del__(self):
        self.close()


@Servers.register("NativeObject", "ros2")
class ROS2NativeObjectServer(ROS2Server):
    SEND_QUEUE = queue.Queue(maxsize=1)
    RECEIVE_QUEUE = queue.Queue(maxsize=1)

    def __init__(self, name, out_port, carrier="", out_port_connect=None, serializer_kwargs=None, deserializer_kwargs=None, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}
        self._plugin_decoder_hook = JsonDecodeHook(**kwargs).object_hook
        self._deserializer_kwargs = deserializer_kwargs or {}

        self._server = None

    def establish(self):
        try:
            from wrapyfi_ros2_interfaces.srv import ROS2NativeObjectService
        except ImportError:
            import wrapyfi
            logging.error("Could not import ROS2NativeObjectService. "
                          "Make sure the ros2 services in wrapyfi_extensions/wrapyfi_ros2_interfaces are compiled. "
                          "Refer to the documentation for more information: \n" +
                          wrapyfi.__url__ + "wrapyfi_extensions/wrapyfi_ros2_interfaces/README.md")
            sys.exit(1)
        self._server = self.create_service(ROS2NativeObjectService, self.out_port, self._service_callback)
        self.established = True

    def await_request(self, *args, **kwargs):
        if not self.established:
            self.establish()
        try:
            request = ROS2NativeObjectServer.RECEIVE_QUEUE.get(block=True)
            [args, kwargs] = json.loads(request.data, object_hook=self._plugin_decoder_hook, **self._deserializer_kwargs)
            return args, kwargs
        except Exception as e:
            logging.error("Service call failed %s" % e)
            return [], {}

    @staticmethod
    def _service_callback(msg):
       ROS2NativeObjectServer.RECEIVE_QUEUE.put(msg)
       return ROS2NativeObjectServer.SEND_QUEUE.get(block=True)

    def reply(self, obj):
        try:
            obj_str = json.dumps(obj, cls=self._plugin_encoder, **self._plugin_kwargs,
                                 serializer_kwrags=self._serializer_kwargs)
            obj_msg = std_msgs.msg.String()
            obj_msg.data = obj_str
            ROS2NativeObjectServer.SEND_QUEUE.put(obj_msg, block=False)
        except queue.Full:
            logging.warning(f"Discarding data because queue is full. "
                            f"This happened due to bad synchronization in {self.__name__}")

