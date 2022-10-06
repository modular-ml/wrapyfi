import logging
import atexit
import threading
import multiprocessing

import zmq

from wrapyfi.utils import SingletonOptimized


class ZeroMQMiddleware(metaclass=SingletonOptimized):

    @staticmethod
    def activate(**kwargs):
        ZeroMQMiddleware(zmq_proxy_kwargs=kwargs)

    def __init__(self, zmq_proxy_kwargs=None, *args, **kwargs):
        logging.info("Initialising ZeroMQ middleware")
        ctx = zmq.Context.instance()
        atexit.register(self.deinit)
        if zmq_proxy_kwargs is not None and zmq_proxy_kwargs:
            if zmq_proxy_kwargs["proxy_broker_spawn"] == "process":
                proxy = multiprocessing.Process(name='zeromq_broker', target=self.__init_proxy, kwargs=zmq_proxy_kwargs)
                proxy.daemon = True
                proxy.start()
            else:  # if threaded
                proxy = threading.Thread(name='zeromq_broker', target=self.__init_proxy, kwargs=zmq_proxy_kwargs)
                proxy.setDaemon(True)
                proxy.start()
            pass

    @staticmethod
    def __init_proxy(socket_address="tcp://*:5555", socket_sub_address="tcp://*:5556",
                     proxy_broker_verbose=False, **kwargs):
        xpub = zmq.Context.instance().socket(zmq.XPUB)
        try:
            xpub.bind(socket_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_address}")
            return
        xsub = zmq.Context.instance().socket(zmq.XSUB)
        try:
            xsub.bind(socket_sub_address)
        except zmq.ZMQError as e:
            logging.error(f"[ZeroMQ] {e} {socket_sub_address}")
            return
        logging.info(f"[ZeroMQ] Intialising proxy broker")
        poller = zmq.Poller()
        poller.register(xpub, zmq.POLLIN)
        poller.register(xsub, zmq.POLLIN)

        while True:
            events = dict(poller.poll(1000))
            if xpub in events:
                message = xpub.recv_multipart()
                if proxy_broker_verbose:
                    logging.info("[ZeroMQ] subscription message: %r" % message[0])
                xsub.send_multipart(message)
            if xsub in events:
                message = xsub.recv_multipart()
                if proxy_broker_verbose:
                    logging.info("[ZeroMQ] subscription message: %r" % message[0])
                xpub.send_multipart(message)

    @staticmethod
    def deinit():
        logging.info("Deinitialising ZeroMQ middleware")
        zmq.Context.instance().destroy()
