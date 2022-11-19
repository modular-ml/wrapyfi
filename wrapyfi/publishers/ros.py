import logging
import sys
import json
import time
import os
import importlib.util

import numpy as np
import rospy
import std_msgs.msg
import sensor_msgs.msg

from wrapyfi.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapyfi.middlewares.ros import ROSMiddleware
from wrapyfi.encoders import JsonEncoder


QUEUE_SIZE = int(os.environ.get("WRAPYFI_ROS_QUEUE_SIZE", 5))
WATCHDOG_POLL_REPEAT = None


class ROSPublisher(Publisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, queue_size=QUEUE_SIZE, ros_kwargs=None, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        ROSMiddleware.activate(**ros_kwargs or {})
        self.queue_size = queue_size

    def await_connection(self, publisher, out_port=None, repeats=None):
        connected = False
        if out_port is None:
            out_port = self.out_port
        logging.info(f"Waiting for topic subscriber: {out_port}")
        if repeats is None:
            if self.should_wait:
                repeats = -1
            else:
                repeats = 1
            while repeats > 0 or repeats <= -1:
                repeats -= 1
                connected = publisher.get_num_connections() < 1
                if connected:
                    break
                time.sleep(0.02)
        logging.info(f"Topic subscriber connected: {out_port}")
        return connected

    def close(self):
        if hasattr(self, "_publisher"):
            self._publisher.shutdown()

    def __del__(self):
        self.close()


@Publishers.register("NativeObject", "ros")
class ROSNativeObjectPublisher(ROSPublisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, queue_size=QUEUE_SIZE, serializer_kwargs=None, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, queue_size=queue_size, **kwargs)
        self._publisher = None

        self._plugin_encoder = JsonEncoder
        self._plugin_kwargs = kwargs
        self._serializer_kwargs = serializer_kwargs or {}

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        self._publisher = rospy.Publisher(self.out_port, std_msgs.msg.String, queue_size=self.queue_size)
        established = self.await_connection(self._publisher, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, obj):
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)
        obj_str = json.dumps(obj, cls=self._plugin_encoder, **self._plugin_kwargs,
                             serializer_kwrags=self._serializer_kwargs)
        self._publisher.publish(obj_str)


@Publishers.register("Image", "ros")
class ROSImagePublisher(ROSPublisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, width=-1, height=-1, rgb=True, fp=False, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        self.width = width
        self.height = height
        self.rgb = rgb
        self.fp = fp
        if self.fp:
            self._encoding = '32FC3' if self.rgb else '32FC1'
            self._type = np.float32
        else:
            self._encoding = 'bgr8' if self.rgb else 'mono8'
            self._type = np.uint8
        self._publisher = None
        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        self._publisher = rospy.Publisher(self.out_port, sensor_msgs.msg.Image, queue_size=self.queue_size)
        established = self.await_connection(self._publisher)
        return self.check_establishment(established)

    def publish(self, img):
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)
        if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or \
                not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
            raise ValueError("Incorrect image shape for publisher")
        img = np.require(img, dtype=self._type, requirements='C')
        msg = sensor_msgs.msg.Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = self._encoding
        msg.is_bigendian = img.dtype.byteorder == '>' or (img.dtype.byteorder == '=' and sys.byteorder == 'big')
        msg.step = img.strides[0]
        msg.data = img.tobytes()
        self._publisher.publish(msg)


@Publishers.register("AudioChunk", "ros")
class ROSAudioChunkPublisher(ROSPublisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, channels=1, rate=44100, chunk=-1, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self._publisher = None
        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, **kwargs):
        self._publisher = rospy.Publisher(self.out_port, sensor_msgs.msg.Image, queue_size=self.queue_size)
        established = self.await_connection(self._publisher)
        return self.check_establishment(established)

    def publish(self, aud):
        if not self.established:
            established = self.establish(repeats=WATCHDOG_POLL_REPEAT)
            if not established:
                return
            else:
                time.sleep(0.2)
        aud, rate = aud
        if rate != self.rate:
            raise ValueError("Incorrect audio rate for publisher")
        if aud is None:
            return
        if 0 < self.chunk != aud.shape[0] or aud.shape[1] != self.channels:
            raise ValueError("Incorrect audio shape for publisher")
        aud = np.require(aud, dtype=np.float32, requirements='C')
        msg = sensor_msgs.msg.Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = aud.shape[0]
        msg.width = aud.shape[1]
        msg.encoding = '32FC1'
        msg.is_bigendian = aud.dtype.byteorder == '>' or (aud.dtype.byteorder == '=' and sys.byteorder == 'big')
        msg.step = aud.strides[0]
        msg.data = aud.tobytes()
        self._publisher.publish(msg)


@Publishers.register("Properties", "ros")
class ROSPropertiesPublisher(ROSPublisher):
    """
    Sets rospy properties. Behaves differently from other data types by directly setting ROS parameters.
    Note that the listener is not guaranteed to receive the updated signal, since the listener can trigger before
    property is set. The property decorated method returns accept native python objects (excluding None), but care should be taken when
    using dictionaries, since they are analogous with node namespaces:
    http://wiki.ros.org/rospy/Overview/Parameter%20Server
    """
    def __init__(self, name, out_port, carrier="", out_port_connect=None, persistent=True, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        self.persistent = persistent

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

        self.previous_property = False

    def establish(self, repeats=-1, **kwargs):
        self.previous_property = rospy.get_param(self.out_port, False)

    def publish(self, obj):
        rospy.set_param(self.out_port, obj)

    def close(self):
        if hasattr(self, "out_port") and not self.persistent:
            rospy.delete_param(self.out_port)
            if self.previous_property:
                rospy.set_param(self.out_port, self.previous_property)

    def __del__(self):
        self.close()


@Publishers.register("ROSMessage", "ros")
class ROSMessagePublisher(ROSPublisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, queue_size=QUEUE_SIZE, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, queue_size=queue_size, **kwargs)
        self._publisher = None

        if not self.should_wait:
            PublisherWatchDog().add_publisher(self)

    def establish(self, repeats=None, obj=None, **kwargs):
        if obj is None:
            return
        obj_type = obj._type.split("/")
        import_msg = importlib.import_module(f"{obj_type[0]}.msg")
        msg_type = getattr(import_msg, obj_type[1])
        self._publisher = rospy.Publisher(self.out_port, msg_type, queue_size=self.queue_size)
        established = self.await_connection(self._publisher, repeats=repeats)
        return self.check_establishment(established)

    def publish(self, obj):
        if not self.established:
            established = self.establish(obj=obj)
            if not established:
                return
            else:
                time.sleep(0.2)
        self._publisher.publish(obj)
