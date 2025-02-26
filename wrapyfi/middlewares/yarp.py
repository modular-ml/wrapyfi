import logging
import atexit

import yarp

from wrapyfi.utils.core_utils import SingletonOptimized
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class YarpMiddleware(metaclass=SingletonOptimized):
    """
    YARP middleware wrapper. This class is a singleton, so it can be instantiated only once. The ``activate`` method
    should be called to initialize the middleware. The ``deinit`` method should be called to deinitialize the middleware
    and destroy all connections. The ``activate`` and ``deinit`` methods are automatically called when the class is
    instantiated and when the program exits, respectively.
    """

    @staticmethod
    def activate(**kwargs):
        """
        Activate the YARP middleware. This method should be called to initialize the middleware.

        :param kwargs: dict: Keyword arguments to be passed to the YARP initialization function
        """
        YarpMiddleware(**kwargs)

    def __init__(self, *args, **kwargs):
        """
        Initialize the YARP middleware. This method is automatically called when the class is instantiated.
        """
        logging.info("Initialising YARP middleware")
        yarp.Network.init()
        atexit.register(MiddlewareCommunicator.close_all_instances)
        atexit.register(self.deinit)

    @staticmethod
    def deinit():
        """
        Deinitialize the YARP middleware. This method is automatically called when the program exits.
        """
        logging.info("Deinitializing YARP middleware")
        yarp.Network.fini()
