import logging
import atexit

import yarp

from wrapyfi.utils import SingletonOptimized


class YarpMiddleware(metaclass=SingletonOptimized):

    @staticmethod
    def activate(**kwargs):
        YarpMiddleware(**kwargs)

    def __init__(self, *args, **kwargs):
        logging.info("Initialising YARP middleware")
        yarp.Network.init()
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        logging.info("Deinitialising YARP middleware")
        yarp.Network.fini()
