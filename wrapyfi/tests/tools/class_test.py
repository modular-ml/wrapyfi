import logging
import time
import numpy as np


from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class Test(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                     should_wait="$should_wait")
    def exchange_object(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_native_exchange", should_wait=False):
        ret = {"message": msg,
               "set": {'a', 1, None},
               "list": [[[3, [4], 5.677890, 1.2]]],
               "string": "string of characters",
               "2": 2.73211,
               "dict": {"other": [None, False, 16, 4.32,]}}
        return ret,

    ######################################## Encoders ########################################

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                     carrier="tcp", should_wait="$should_wait")
    def exchange_object_jax(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_jax", should_wait=False):
        try:
            import jax.numpy as jnp
        except ImportError:
            logging.warning("jax not installed")
            return
        msg = input("Type your message: ")
        ret = {"message": msg,
               "jax_ones": jnp.ones((2, 4))}
        return ret,

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                        carrier="tcp", should_wait="$should_wait")
    def exchange_object_torch(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_torch", should_wait=False):
        try:
            import torch
        except ImportError:
            logging.warning("torch not installed")
            return
        msg = input("Type your message: ")
        ret = {"message": msg,
               "torch_ones": torch.ones((2, 4))}
        return ret,

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                     carrier="tcp", should_wait="$should_wait")
    def exchange_object_tensorflow(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_tensorflow", should_wait=False):
        try:
            import tensorflow as tf
        except ImportError:
            logging.warning("tensorflow not installed")
            return
        msg = input("Type your message: ")
        ret = {"message": msg,
               "tensorflow_ones": tf.ones((2, 4))}
        return ret,

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                     carrier="tcp", should_wait="$should_wait")
    def exchange_object_paddlepadde(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_paddlepadde", should_wait=False):
        try:
            import paddle
        except ImportError:
            logging.warning("paddlepadde not installed")
            return
        msg = input("Type your message: ")
        ret = {"message": msg,
               "paddlepadde_ones": paddle.ones((2, 4))}
        return ret,

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                        carrier="tcp", should_wait="$should_wait")
    def exchange_object_mxnet(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_mxnet", should_wait=False):
        try:
            import mxnet
        except ImportError:
            logging.warning("mxnet not installed")
            return
        msg = input("Type your message: ")
        ret = {"message": msg,
               "mxnet_ones": mxnet.ones((2, 4))}
        return ret,

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic", carrier="tcp", should_wait="$should_wait")
    def exchange_object_pyarrow(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_pyarrow", should_wait=False):
        try:
            import pyarrow
        except ImportError:
            logging.warning("pyarrow not installed")
            return
        msg = input("Type your message: ")
        ret = {"message": msg,
               "pyarrow_ones": pyarrow.ones((2, 4))}
        return ret,

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                            carrier="tcp", should_wait="$should_wait")
    def exchange_object_numpy(msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_numpy", should_wait=False):
        ret = {"message": msg,
               "numpy_ones": np.ones((2, 4))}
        return ret,

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                        carrier="tcp", should_wait="$should_wait")
    def exchange_object_pandas(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_pandas", should_wait=False):
        try:
            import pandas as pd
        except ImportError:
            logging.warning("pandas not installed")
            return
        msg = input("Type your message: ")
        ret = {"message": msg,
               "pandas_ones": pd.DataFrame(np.ones((2, 4)))}
        return ret,

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                        carrier="tcp", should_wait="$should_wait")
    def exchange_object_dask(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_dask", should_wait=False):
        try:
            import dask.array as da
        except ImportError:
            logging.warning("dask not installed")
            return
        msg = input("Type your message: ")
        ret = {"message": msg,
               "dask_ones": da.ones((2, 4))}
        return ret,

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                        carrier="tcp", should_wait="$should_wait")
    def exchange_object_xarray(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_xarray", should_wait=False):
        try:
            import xarray as xr
        except ImportError:
            logging.warning("xarray not installed")
            return
        msg = input("Type your message: ")
        ret = {"message": msg,
               "xarray_ones": xr.ones((2, 4))}
        return ret,


def test_func(queue_buffer, mode="listen", mware=DEFAULT_COMMUNICATOR, topic="/test/test_native_exchange", iterations=2,
              should_wait=False):
    test = Test()
    test.activate_communication(test.exchange_object, mode=mode)
    for i in range(iterations):
        my_message = test.exchange_object(msg=f"signal_idx:{i}", mware=mware, topic=topic, should_wait=should_wait)
        if my_message is not None:
            print(f"result {mode}:", my_message[0]["message"])
            queue_buffer.put(my_message[0])
        time.sleep(0.5)