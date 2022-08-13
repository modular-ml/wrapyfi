import atexit
import yarp

from wrapify.utils import SingletonOptimized


class YarpMiddleware(metaclass=SingletonOptimized):

    @staticmethod
    def activate():
        YarpMiddleware()

    def __init__(self):
        print("Initialising YARP middleware")
        yarp.Network.init()
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        print("Deinitialising YARP middleware")
        yarp.Network.fini()
