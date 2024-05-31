import argparse
import time

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Install pandas and NumPy before running this script.")
try:
    import tensorflow as tf

    # avoid allocating all GPU memory assuming tf>=2.2
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except ImportError:
    tf = None

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


SHOULD_WAIT = True


class Benchmarker(MiddlewareCommunicator):

    @staticmethod
    def get_numpy_object(dims):
        return {"numpy": np.ones(dims)}

    @staticmethod
    def get_pandas_object(dims):
        return {
            "pandas": pd.DataFrame(
                np.ones(dims), index=None, columns=list(range(dims[-1]))
            )
        }
        
    @staticmethod
    def get_cupy_gpu_object(dims, gpu=0):
        import cupy as cp
        with cp.cuda.Device(gpu):
            cp_ones = cp.ones(dims, dtype=cp.float32)
        return {"cupy_gpu": cp_ones}
    
    @staticmethod
    def get_pyarrow_object(dims):
        import pyarrow as pa
        return {"pyarrow": pa.array(np.ones(dims).flatten())}    
    
    @staticmethod
    def get_xarray_object(dims):
        import xarray as xr
        return {"xarray": xr.DataArray(np.ones(dims), name="example")}
    
    @staticmethod
    def get_dask_object(dims):
        import dask.array as da
        return {"dask": da.ones(dims)}
        
    @staticmethod
    def get_pillow_object(dims):
        from PIL import Image
        return {"pillow": Image.fromarray(np.ones(dims, dtype=np.uint8))}

    @staticmethod
    def get_tensorflow_object(dims):
        return {"tensorflow": tf.ones(dims)}

    @staticmethod
    def get_jax_object(dims):
        import jax as jx
        return {"jax": jx.numpy.ones(dims)}

    @staticmethod
    def get_mxnet_object(dims):
        import mxnet as mx
        return {"mxnet": mx.nd.ones(dims)}

    @staticmethod
    def get_mxnet_gpu_object(dims, gpu=0):
        import mxnet as mx
        return {"mxnet_gpu": mx.nd.ones(dims, ctx=mx.gpu(gpu))}

    @staticmethod
    def get_pytorch_object(dims):
        import torch as th
        return {"pytorch": th.ones(dims)}

    @staticmethod
    def get_pytorch_gpu_object(dims, gpu=0):
        import torch as th
        return {"pytorch_gpu": th.ones(dims, device=f"cuda:{gpu}")}

    @staticmethod
    def get_paddle_object(dims):
        import paddle as pa
        return {"paddle": pa.Tensor(pa.ones(dims), place=pa.CPUPlace())}

    @staticmethod
    def get_paddle_gpu_object(dims, gpu=0):
        import paddle as pa
        return {"paddle_gpu": pa.Tensor(pa.zeros(dims), place=pa.CUDAPlace(gpu))}
        
    def get_all_objects(self, count, plugin_name):
        obj = {"count": count, "timestamp": time.time()}
        object_creator = getattr(self, f"get_{plugin_name}_object")
        data_object = object_creator((args.height, args.width))
        if plugin_name == "dask":
            data_object["dask"] = data_object["dask"].compute()
        obj.update(data_object)
        return obj

    @MiddlewareCommunicator.register(
        "NativeObject",
        "yarp",
        "ExampleClass",
        "/example/get_native_objects",
        carrier="tcp",
        should_wait=SHOULD_WAIT,
    )
    def get_yarp_native_objects(self, count, plugin_name):
        return (self.get_all_objects(count, plugin_name),)

    @MiddlewareCommunicator.register(
        "NativeObject",
        "ros",
        "ExampleClass",
        "/example/get_native_objects",
        carrier="tcp",
        should_wait=SHOULD_WAIT,
    )
    def get_ros_native_objects(self, count, plugin_name):
        return (self.get_all_objects(count, plugin_name),)

    @MiddlewareCommunicator.register(
        "NativeObject",
        "ros2",
        "ExampleClass",
        "/example/get_native_objects",
        carrier="tcp",
        should_wait=SHOULD_WAIT,
    )
    def get_ros2_native_objects(self, count, plugin_name):
        return (self.get_all_objects(count, plugin_name),)

    @MiddlewareCommunicator.register(
        "NativeObject",
        "zeromq",
        "ExampleClass",
        "/example/get_native_objects",
        carrier="tcp",
        should_wait=SHOULD_WAIT,
    )
    def get_zeromq_native_objects(self, count, plugin_name):
        return (self.get_all_objects(count, plugin_name),)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publish",
        dest="mode",
        action="store_const",
        const="publish",
        default="listen",
        help="Publish mode",
    )
    parser.add_argument(
        "--listen",
        dest="mode",
        action="store_const",
        const="listen",
        default="listen",
        help="Listen mode (default)",
    )
    parser.add_argument(
        "--mwares",
        type=str,
        default=list(MiddlewareCommunicator.get_communicators()),
        choices=MiddlewareCommunicator.get_communicators(),
        nargs="+",
        help="The middlewares to use for transmission",
    )
    parser.add_argument(
        "--plugins",
        type=str,
        default=[
            "numpy",
            "pandas",
            "cupy_gpu",
            "pyarrow",
            "xarray",
            "dask",
            "tensorflow",
            "jax",
            "mxnet",
            "mxnet_gpu",
            "pytorch",
            "pytorch_gpu",
            "paddle",
            "paddle_gpu",
        ],
        nargs="+",
        help="The middlewares to use for transmission",
    )
    parser.add_argument(
        "--height", type=int, default=200, help="The tensor image height"
    )
    parser.add_argument("--width", type=int, default=200, help="The tensor image width")
    parser.add_argument(
        "--trials",
        type=int,
        default=2000,
        help="Number of trials to run per middleware",
    )
    parser.add_argument(
        "--skip-trials",
        type=int,
        default=0,
        help="Number of trials to skip before logging "
        "to csv to avoid warmup time logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmarker = Benchmarker()
    benchmark_logger = pd.DataFrame(
        columns=["middleware", "plugin", "timestamp", "count", "delay"]
    )
    benchmark_iterator = {}

    for middleware_name in args.mwares:
        benchmark_iterator[middleware_name] = getattr(
            benchmarker, f"get_{middleware_name}_native_objects"
        )

    for middleware_name, method in benchmark_iterator.items():
        benchmarker.activate_communication(method, mode=args.mode)
        for plugin_name in args.plugins:
            time_acc_native_objects = []
            counter = -1
            while True:
                counter += 1
                (native_objects,) = method(counter, plugin_name)
                if native_objects is not None:
                    time_acc_native_objects.append(time.time() - native_objects["timestamp"])
                    print(
                        f"{middleware_name} :: {plugin_name} :: delay:",
                        time_acc_native_objects[-1],
                        " Length:",
                        len(time_acc_native_objects),
                        " Count:",
                        native_objects["count"],
                    )
                    if args.trials - 1 == native_objects["count"]:
                        break
                    if counter > args.skip_trials:
                        benchmark_logger = benchmark_logger.append(
                            pd.DataFrame(
                                {
                                    "middleware": [middleware_name],
                                    "plugin": [plugin_name],
                                    "timestamp": [native_objects["timestamp"]],
                                    "count": [native_objects["count"]],
                                    "delay": [time_acc_native_objects[-1]],
                                }
                            ),
                            ignore_index=True,
                        )
                    if counter == 0:
                        if args.mode == "publish":
                            time.sleep(5)
                    else:
                        if args.mode == "publish":
                            time.sleep(0.1)

            time_acc_native_objects = pd.DataFrame(np.array(time_acc_native_objects))
            print(
                f"{middleware_name} :: {plugin_name} :: time statistics:",
                time_acc_native_objects.describe(),
            )
            time.sleep(5)
    benchmark_logger.to_csv(
        f"results/benchmarking_native_object_{args.mode}__{','.join(args.mwares)}__{','.join(args.plugins)}.csv",
        index=False,
    )
