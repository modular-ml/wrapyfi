import sys
import json
import time
import numpy as np
import rospy
import std_msgs.msg
import sensor_msgs.msg

from wrapify.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapify.middlewares.ros import ROSMiddleware
from wrapify.utils import JsonEncoder


class ROSPublisher(Publisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, queue_size=5, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        ROSMiddleware.activate()
        self.queue_size = queue_size

    def await_connection(self, publisher, out_port=None):
        if out_port is None:
            out_port = self.out_port
        print("Waiting for topic subscriber:", out_port)
        while publisher.get_num_connections() < 1:
            time.sleep(0.02)
        print("Topic subscriber connected:", out_port)


@Publishers.register("NativeObject", "ros")
class ROSNativeObjectPublisher(ROSPublisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, queue_size=5, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, queue_size=queue_size, **kwargs)
        self._publisher = None
        PublisherWatchDog().add_publisher(self)

    def establish(self):
        self._publisher = rospy.Publisher(self.out_port, std_msgs.msg.String, queue_size=self.queue_size)
        self.await_connection(self._publisher)
        self.established = True

    def publish(self, obj):
        if not self.established:
            self.establish()
        obj_str = json.dumps(obj, cls=JsonEncoder)
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
        PublisherWatchDog().add_publisher(self)

    def establish(self):
        self._publisher = rospy.Publisher(self.out_port, sensor_msgs.msg.Image, queue_size=self.queue_size)
        self.await_connection(self._publisher)
        self.established = True

    def publish(self, img):
        if not self.established:
            self.establish()
        if 0 < self.width != img.shape[1] or 0 < self.height != img.shape[0] or not ((img.ndim == 2 and not self.rgb) or (img.ndim == 3 and self.rgb and img.shape[2] == 3)):
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
        PublisherWatchDog().add_publisher(self)

    def establish(self, **kwargs):
        self._publisher = rospy.Publisher(self.out_port, sensor_msgs.msg.Image, queue_size=self.queue_size)
        self.await_connection(self._publisher)
        self.established = True

    def publish(self, aud):
        if not self.established:
            self.establish()
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

    def __init__(self, name, out_port, **kwargs):
        super().__init__(name, out_port, **kwargs)
        raise NotImplementedError
