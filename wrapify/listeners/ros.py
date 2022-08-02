import rospy

from wrapify.connect.listeners import Listener, ListenerWatchDog, Listeners


@Listeners.register("NativeObject", "ros")
class ROSNativeObjectListener(Listener):

    def __init__(self, name, in_port, carrier="", should_wait=False, load_torch_device=None):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)
        # TODO: Initialise variables (e.g. subscriber = None)
        ListenerWatchDog().add_listener(self)

    def establish(self):
        # TODO: Create ROS subscriber
        self.established = True

    def listen(self):
        if not self.established:
            self.establish()
        # TODO: Ctrl+C-able wait for data (or None) depending on self.should_wait
        return None


@Listeners.register("Image", "ros")
class ROSImageListener(Listener):

    def __init__(self, name, in_port, carrier="", should_wait=False, width=320, height=240, rgb=True):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)
        self.width = width
        self.height = height
        self.rgb = rgb
        ListenerWatchDog().add_listener(self)
        raise NotImplementedError


@Listeners.register("AudioChunk", "ros")
class ROSAudioChunkListener(Listener):

    def __init__(self, name, in_port, carrier="", should_wait=False, channels=1, rate=44100, chunk=-1):
        super().__init__(name, in_port, carrier=carrier, should_wait=should_wait)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        ListenerWatchDog().add_listener(self)
        raise NotImplementedError


@Listeners.register("Properties", "ros")
class ROSPropertiesListener(Listener):

    def __init__(self, name, in_port, **kwargs):
        super().__init__(name, in_port, **kwargs)
        raise NotImplementedError
