import rospy

from wrapify.connect.publishers import Publisher, Publishers, PublisherWatchDog


@Publishers.register("NativeObject", "ros")
class ROSNativeObjectPublisher(Publisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect)
        # TODO: Initialise variables (e.g. publisher = None)
        PublisherWatchDog().add_publisher(self)

    def establish(self):
        # TODO: Create ROS publisher
        self.established = True

    def publish(self, obj):
        if not self.established:
            self.establish()
        # TODO: Publish obj to topic


@Publishers.register("Image", "ros")
class ROSImagePublisher(Publisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, width=320, height=240, rgb=True, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect)
        self.width = width
        self.height = height
        self.rgb = rgb
        PublisherWatchDog().add_publisher(self)
        raise NotImplementedError


@Publishers.register("AudioChunk", "ros")
class ROSAudioChunkPublisher(Publisher):

    def __init__(self, name, out_port, carrier="", out_port_connect=None, channels=1, rate=44100, chunk=-1, **kwargs):
        super().__init__(name, out_port, carrier=carrier, out_port_connect=out_port_connect)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        PublisherWatchDog().add_publisher(self)
        raise NotImplementedError


@Publishers.register("Properties", "ros")
class ROSPropertiesPublisher(Publisher):

    def __init__(self, name, out_port, **kwargs):
        super().__init__(name, out_port, **kwargs)
        raise NotImplementedError
