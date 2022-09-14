import argparse
import time

import torch as th
import tensorflow as tf
import mxnet as mx
import numpy as np
import pandas as pd

from wrapify.connect.wrapper import MiddlewareCommunicator


TRIALS = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--publish", dest="mode", action="store_const", const="publish", default="listen", help="Publish mode")
parser.add_argument("--listen", dest="mode", action="store_const", const="listen", default="listen", help="Listen mode (default)")
parser.add_argument("--mwares", type=str, default=["yarp", "ros"], choices={"yarp", "ros"}, nargs="+", help="The middlewares to use for transmission")
args = parser.parse_args()


class Benchmarker(MiddlewareCommunicator):

    def get_dummy_object(self, count):
        obj = {"np": np.ones((200, 200)),
               "tf": tf.ones((200, 200)),
               "mx": mx.nd.ones((200, 200)),
               "th": th.ones((200, 200)),
               "count": count,
               "time": time.time()}
        return obj

    @MiddlewareCommunicator.register("NativeObject", "yarp",
                                     "ExampleClass", "/example/get_native_objects",
                                     carrier="tcp", should_wait=True)
    def yarp_get_native_objects(self, count):
        return self.get_dummy_object(count),

    @MiddlewareCommunicator.register("NativeObject", "ros",
                                     "ExampleClass", "/example/get_native_objects",
                                     carrier="tcp", should_wait=True)
    def ros_get_native_objects(self, count):
        return self.get_dummy_object(count),


benchmarker = Benchmarker()
benchmark_logger = pd.DataFrame(columns=["middleware", "time", "count", "delay"])
benchmark_iterator = {}
if "ros" in args.mwares:
    benchmark_iterator["ros"] = benchmarker.ros_get_native_objects
if "yarp" in args.mwares:
    benchmark_iterator["yarp"] = benchmarker.yarp_get_native_objects


for middleware, method in benchmark_iterator.items():
    benchmarker.activate_communication(method, mode=args.mode)

    time_acc_native_objects = []
    counter = -1
    while True:
        counter += 1
        native_objects, = method(counter)
        if native_objects is not None:
            time_acc_native_objects.append(time.time() - native_objects["time"])
            print(f"{middleware} Native Objects delay:", time_acc_native_objects[-1],
                  " Length:", len(time_acc_native_objects), " Count:", native_objects["count"])
            if TRIALS-1 == native_objects["count"]:
                break
        if counter == 0:
            time.sleep(5)
        else:
            time.sleep(0.1)
        benchmark_logger = benchmark_logger.append(pd.DataFrame({"middleware": [middleware],
                                                                 "time": [native_objects["time"]],
                                                                 "count": [native_objects["count"]],
                                                                 "delay": [time_acc_native_objects[-1]]}),
                                                   ignore_index=True)

    time_acc_native_objects = pd.DataFrame(np.array(time_acc_native_objects))
    print(f"{middleware} Native Objects time statistics:", time_acc_native_objects.describe())
    time.sleep(5)
benchmark_logger.to_csv(f"results/benchmarking_native_object_{args.mode}.csv", index=False)