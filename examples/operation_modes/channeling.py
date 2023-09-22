import argparse
from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")
parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                    help="The middleware to use for transmission")
args = parser.parse_args()


class MirroringClass(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", "yarp",
                                     "ExampleClass", "/example/read_message",
                                     carrier="tcp", should_wait=True)
    def read_message(self):
        msg = input("Type your message: ")
        obj = {"message": msg}
        return obj,


class ChannelingClass(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", "yarp",
                                     "ExampleClass", "/example/native_yarp_message",
                                     carrier="mcast", should_wait=True)
    @MiddlewareCommunicator.register("Image", "ros",
                                     "ExampleClass", "/example/image_ros_message",
                                     carrier="tcp", width="$img_width",
                                     height="$img_height", rgb=True, queue_size=10)
    @MiddlewareCommunicator.register("AudioChunk", "zeromq",
                                     "ExampleClass", "/example/audio_zeromq_image",
                                     carrier="tcp", rate="$aud_rate",
                                     chunk="$aud_chunk", channels="$aud_channels")
    def read_mulret_mulmware(self, img_width=200, img_height=200,
                             aud_rate=44100, aud_chunk=8820, aud_channels=1):
        ros_img = np.random.randint(256, size=(img_height, img_width, 3),
                                     dtype=np.uint8)
        zeromq_aud = np.random.uniform(-1, 1, aud_chunk)
        yarp_native = [ros_img, zeromq_aud]
        return yarp_native, ros_img, zeromq_aud


channeling = ChannelingClass()
channeling.activate_communication(ChannelingClass.read_mulret_mulmware, mode=args.mode)

while True:
    yarp_native, ros_img, zeromq_aud = channeling.read_mulret_mulmware()
    if yarp_native is not None:
        print("yarp_native::", yarp_native)
    if ros_img is not None:
        print("ros_img::", ros_img)
    if zeromq_aud is not None:
        print("zeromq_aud", zeromq_aud)
