import logging
import atexit

import zmq

from wrapify.utils import SingletonOptimized


class ZeroMQMiddleware(metaclass=SingletonOptimized):

    @staticmethod
    def activate():
        ZeroMQMiddleware()

    def __init__(self):
        logging.info("Initialising ZeroMQ middleware")
        zmq.Context.instance()
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        logging.info("Deinitialising ZeroMQ middleware")
        zmq.Context.instance().term()
