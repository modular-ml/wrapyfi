import json
import rospy
import std_msgs.msg

from wrapify.connect.publishers import Publisher, Publishers, PublisherWatchDog
from wrapify.middlewares.ros import ROSMiddleware
from wrapify.utils import JsonEncoder


class ROSPublisher(Publisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, queue_size=5, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, **kwargs)
        ROSMiddleware.activate()
        self.queue_size = queue_size


@Publishers.register("NativeObject", "ros")
class ROSNativeObjectPublisher(ROSPublisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, queue_size=5, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect, queue_size=queue_size, **kwargs)
        self._publisher = None
        PublisherWatchDog().add_publisher(self)

    def establish(self):
        self._publisher = rospy.Publisher(self.out_port, std_msgs.msg.String, queue_size=self.queue_size)
        self.established = True

    def publish(self, obj):
        if not self.established:
            self.establish()
        obj_str = json.dumps(obj, cls=JsonEncoder)
        self._publisher.publish(obj_str)


@Publishers.register("Image", "ros")
class ROSImagePublisher(ROSPublisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, width=320, height=240, rgb=True, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect)
        self.width = width
        self.height = height
        self.rgb = rgb
        PublisherWatchDog().add_publisher(self)
        raise NotImplementedError


@Publishers.register("AudioChunk", "ros")
class ROSAudioChunkPublisher(ROSPublisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, channels=1, rate=44100, chunk=-1, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        PublisherWatchDog().add_publisher(self)
        raise NotImplementedError


@Publishers.register("Properties", "ros")
class ROSPropertiesPublisher(ROSPublisher):

    def __init__(self, name, out_port, **kwargs):
        super().__init__(name, out_port, **kwargs)
        raise NotImplementedError
