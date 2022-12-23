import argparse
import time

import torch as th
import mxnet as mx
import paddle as pa
import jax as jx
import numpy as np
import pandas as pd
import tensorflow as tf
# avoid allocating all GPU memory assuming tf>=2.2
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


SHOULD_WAIT = False

parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")
parser.add_argument("--mwares", type=str, default=["ros", "yarp", "zeromq"],  # ros2
                    choices=MiddlewareCommunicator.get_communicators(), nargs="+",
                    help="The middlewares to use for transmission")
parser.add_argument("--plugins", type=str,
                    default=["numpy", "pandas", "tensorflow", "jax", "mxnet", "mxnet_gpu", "pytorch", "pytorch_gpu",
                             "paddle", "paddle_gpu"], nargs="+",
                    help="The middlewares to use for transmission")
parser.add_argument("--height", type=int, default=200, help="The tensor image height")
parser.add_argument("--width", type=int, default=200, help="The tensor image width")
parser.add_argument("--trials", type=int, default=2000, help="Number of trials to run per middleware")
parser.add_argument("--skip-trials", type=int, default=0, help="Number of trials to skip before logging "
                                                                  "to csv to avoid warmup time logging")
args = parser.parse_args()


class Benchmarker(MiddlewareCommunicator):

    def get_numpy_object(self, dims):
        return {"numpy": np.ones(dims)}

    def get_pandas_object(self, dims):
        return {"pandas": pd.DataFrame(np.ones(dims), index=None, columns=list(range(dims[-1])))}
    def get_tensorflow_object(self, dims):
        return {"tensorflow": tf.ones(dims)}

    def get_jax_object(self, dims):
        return {"jax": jx.numpy.ones(dims)}

    def get_mxnet_object(self, dims):
        return {"mxnet": mx.nd.ones(dims)}

    def get_mxnet_gpu_object(self, dims, gpu=0):
        return {"mxnet_gpu": mx.nd.ones(dims, ctx=mx.gpu(gpu))}

    def get_pytorch_object(self, dims):
        return {"pytorch": th.ones(dims)}

    def get_pytorch_gpu_object(self, dims, gpu=0):
        return {"pytorch_gpu": th.ones(dims, device=f"cuda:{gpu}")}

    def get_paddle_object(self, dims):
        return {"paddle": pa.Tensor(pa.ones(dims), place=pa.CPUPlace())}

    def get_paddle_gpu_object(self, dims, gpu=0):
        return {"paddle_gpu": pa.Tensor(pa.zeros(dims), place=pa.CUDAPlace(gpu))}


    def get_all_objects(self, count, plugin_name):
        obj = {"count": count,
               "time": time.time()}
        obj.update(**getattr(self, f"get_{plugin_name}_object")((args.height, args.width,)))
        return obj

    @MiddlewareCommunicator.register("NativeObject", "yarp",
                                     "ExampleClass", "/example/get_native_objects",
                                     carrier="tcp", should_wait=SHOULD_WAIT)
    def yarp_get_native_objects(self, count, plugin_name):
        return self.get_all_objects(count, plugin_name),

    @MiddlewareCommunicator.register("NativeObject", "ros",
                                     "ExampleClass", "/example/get_native_objects",
                                     carrier="tcp", should_wait=SHOULD_WAIT)
    def ros_get_native_objects(self, count, plugin_name):
        return self.get_all_objects(count, plugin_name),

    @MiddlewareCommunicator.register("NativeObject", "ros2",
                                     "ExampleClass", "/example/get_native_objects",
                                     carrier="tcp", should_wait=SHOULD_WAIT)
    def ros2_get_native_objects(self, count, plugin_name):
        return self.get_all_objects(count, plugin_name),

    @MiddlewareCommunicator.register("NativeObject", "zeromq",
                                     "ExampleClass", "/example/get_native_objects",
                                     carrier="tcp", should_wait=SHOULD_WAIT)
    def zeromq_get_native_objects(self, count, plugin_name):
        return self.get_all_objects(count, plugin_name),


benchmarker = Benchmarker()
benchmark_logger = pd.DataFrame(columns=["middleware", "plugin", "time", "count", "delay"])
benchmark_iterator = {}


if "ros" in args.mwares:
    benchmark_iterator["ros"] = benchmarker.ros_get_native_objects
if "ros2" in args.mwares:
    benchmark_iterator["ros2"] = benchmarker.ros2_get_native_objects
if "zeromq" in args.mwares:
        benchmark_iterator["zeromq"] = benchmarker.zeromq_get_native_objects
if "yarp" in args.mwares:
    benchmark_iterator["yarp"] = benchmarker.yarp_get_native_objects

for middleware_name, method in benchmark_iterator.items():
    benchmarker.activate_communication(method, mode=args.mode)
    for plugin_name in args.plugins:
        time_acc_native_objects = []
        counter = -1
        while True:
            counter += 1
            native_objects, = method(counter, plugin_name)
            if native_objects is not None:
                time_acc_native_objects.append(time.time() - native_objects["time"])
                print(f"{middleware_name} :: {plugin_name} :: delay:", time_acc_native_objects[-1],
                      " Length:", len(time_acc_native_objects), " Count:", native_objects["count"])
                if args.trials - 1 == native_objects["count"]:
                    break
                if counter > args.skip_trials:
                    benchmark_logger = benchmark_logger.append(pd.DataFrame({"middleware": [middleware_name],
                                                                             "plugin": [plugin_name],
                                                                             "time": [native_objects["time"]],
                                                                             "count": [native_objects["count"]],
                                                                             "delay": [time_acc_native_objects[-1]]}),
                                                               ignore_index=True)
                if counter == 0:
                    if args.mode == "publish":
                        time.sleep(5)
                else:
                    if args.mode == "publish":
                        time.sleep(0.1)

        time_acc_native_objects = pd.DataFrame(np.array(time_acc_native_objects))
        print(f"{middleware_name} :: {plugin_name} :: time statistics:", time_acc_native_objects.describe())
        time.sleep(5)
benchmark_logger.to_csv(f"results/benchmarking_native_object_{args.mode}__{','.join(args.mwares)}__{','.join(args.plugins)}.csv", index=False)