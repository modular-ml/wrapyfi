# Usage

To Wrapyfi your code:

```
from wrapyfi.connect.wrapper import MiddlewareCommunicator

class TheClass(MiddlewareCommunicator)
        ...
           
        @MiddlewareCommunicator.register(...)
        @MiddlewareCommunicator.register(...)
        def encapsulated_func(...):
            ...
            return encapsulated_a, encapsulated_b
        
        def encapsulating_func(...)
            ...
            encapsulated_a, encapsulated_b = self.encapsulated_func(...)
            ...
            return result,


the_class = TheClass()
the_class.activate_communication(the_class.encapsulated_func, mode="publish")
while True:
    the_class.encapsulating_func(...)
```

The primary component for facilitating communication is the `MiddlewareCommunicator`. To register the 
functions for a given class, it should inherit the `MiddlewareCommunicator`. Any function decorated with
`@MiddlewareCommunicator.register(<Data structure type>, <Communicator>, <Class name>, <Port name>)`. 

The `<Data structure type>` is the publisher/listener type for a given function's return. The supported data
types are listed in the [publishers and listeners](#publishers-and-listeners) section.

The `<Communicator>` defines the communication medium e.g.: `yarp`, `ros2`, `ros`, or `zeromq`. The default communicator is `zeromq` but can be replaced by setting the environment variables `WRAPYFI_DEFAULT_COMMUNICATOR` or `WRAPYFI_DEFAULT_MWARE` (`WRAPYFI_DEFAULT_MWARE` overrides `WRAPYFI_DEFAULT_COMMUNICATOR` when both are provided) to the middleware of choice e.g.: 
        
```
        export WRAPYFI_DEFAULT_COMMUNICATOR=yarp
```

The `<Class name>` serves no purpose in the current Wrapyfi version, but has been left for future support of module-level decoration, 
where the functions don't belong to a class, and must therefore have a unique identifier for declaration in the 
[configuration files](#configuration).

The `<Port name>` is the name used for the connected port and is dependent on the Middleware platform. The listener and publisher receive 
the same port name.

The `@MiddlewareCommunicator.register` decorator is defined for each of the function's returns in the 
same order. As shown in the example above, the first decorator defines the properties of `encapsulated_a`'s 
publisher and listener, whereas the second decorator belongs to `encapsulated_b`. A decorated function must always return a tuple which can easily
be enforced by adding a `comma` after the returning in case a single variable is returned. Lists are also supported for 
single returns e.g.:
```
        @MiddlewareCommunicator.register([..., {...}], [..., {...}], [...])
        @MiddlewareCommunicator.register(...)
        def encapsulated_func(...):
            ...
            encapsulated_a = [[...], [...], [...]]
            ...
            return encapsulated_a, encapsulated_b
```
Each of the list's returns are encapsulated with its own publisher and listener, with the named arguments transmitted as 
a single dictionary within the list. Notice that `encapsulated_a` returns a list of length 3, therefore, the first decorator contains 
3 list configurations as well. Note that by using a single `NativeObject` as a `<Data structure type>`, the same 
can be achieved. However, the Yarp implementation of the `NativeObject` utilizes `BufferedPortBottle` and serializes the 
object before transmission. The `NativeObject` may result in a greater overhead and should only be used when multiple nesting depths are 
required or the objects within a list not within the [supported data structure types](#publishers-and-listeners).

## Configuration
The `MiddlewareCommunicator`'s child class' functions modes can be independently set to:
* **publish**: Run the function and publish the results using the middleware's transmission protocol
* **listen**: Skip the function and wait for the publisher with the same port name to transmit a message, eventually returning the received message
* **none**(default): Run the function as usual without triggering publish or listen
* **disable**: Disables the function and returns None for all its returns. Caution should be taken when disabling a function since it 
could break subsequent calls

These properties can be set by calling: 

`activate_communication(<Function name>, mode=<Mode>)` 

for each docorated function within the class. This however requires modifying your scripts for each machine or process running
on Wrapyfi. To overcome this limitation, use the `ConfigManager` e.g.:
```
from wrapyfi.config.manager import ConfigManager
ConfigManager(<Configuration file path *.yml>)
``` 

The `ConfigManager` is a singleton which must be called once before the initialization of any `MiddlewareCommunicator`. Initializing it 
multiple times has no effect. This limitation was created by design to avoid loading the configuration file multiple times.

The `<Configuration file path *.yml>`'s configuration file has a very simple format e.g.:
```
TheClass:
  encapsulated_func: "publish"

```
Where `TheClass` is the class name, `encapsulated_func` is the function's name and `publish` is the transmission mode.
This is useful when running the same script on multiple machines, where one is set to publish and the other listens.

## Publishers and Listeners

The publishers and listeners of the same message type should have identical constructor signatures. The current Wrapyfi version supports
4 types of messages 

(Yarp):

* **Image**: Transmits and receives a `cv2` or `numpy` image using either `yarp.BufferedPortImageRgb` or `yarp.BufferedPortImageFloat`
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `yarp.BufferedPortImageFloat`, concurrently transmitting the sound properties using `yarp.BufferedPortSound`
* **NativeObject**: Transmits a `json` string supporting all native python objects, `numpy` arrays and [other formats](#data-formats) using `yarp.BufferedPortBottle`
* **Properties**: Transmits properties [*coming soon*]

(ROS):

* **Image**: Transmits and receives a `cv2` or `numpy` image using `rospy sensor_messages.msg.Image`
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `rospy sensor_messages.msg.Image`
* **NativeObject**: Transmits a string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `rospy std_msgs.msg.String`
* **Properties**: Transmits properties [*coming soon*]

(ROS 2): 

* **Image**: Transmits and receives a `cv2` or `numpy` image using `rospy sensor_messages.msg.Image`
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `rospy sensor_messages.msg.Image`
* **NativeObject**: Transmits a string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `rospy std_msgs.msg.String`
* **Properties**: Transmits properties [*coming soon*]

(ZeroMQ):
* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct.
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct.
* **NativeObject**: Transmits a string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `zmq context.socket(zmq.PUB).send_multipart`
* **Properties**: Transmits properties [*coming soon*]

### Publisher- and Listener-specific Arguments

To direct arguments specifically toward the publisher or subscriber without exposing one or the other to the same argument values, the corresponding arguments can be added to the dictionary `listener_kwargs` to control the listener only, or `publisher_kwargs` to control the publisher only. Both dictionaries can be passed directly to the Wrapyfi decorator.

### Plugins

The **NativeObject** message type supports structures beyond native python objects. Wrapyfi already supports a number of non-native objects including numpy arrays and tensors. Wrapyfi can be extended to support objects by using the plugins API. All currently supported plugins by Wrapyfi can be found in the [plugins directory](../wrapyfi/plugins). Plugins can be added by:
* Creating a derived class that inherits from the base class `wrapify.utils.Plugin`
* Overriding the `encode` method for converting the object to a JSON serializable string. Deserializing the string is performed within the overridden `decode` method.
* Specifying custom object properties by defining keyword arguments for the class constructor. These properties can be passed directly to the Wrapyfi decorator
* Decorating the class with `@PluginRegistrar.register` appending the plugin to the list of supported objects
* Appending the script path where the class is defined to the `WRAPYFI_PLUGINS_PATH` environment variable
* Ensure that the plugin resides within a directory named `plugins` residing inside the `WRAPYFI_PLUGINS_PATH` and that the directory contains an `__init__.py` file
    
#### Data Structure Types

Other than native python objects, the following objects are supported:

* `numpy.ndarray` and `numpy.generic`
* `pandas.DataFrame` and `pandas.Series`
* `torch.Tensor`
* `tensorflow.Tensor` and `tensorflow.EagerTensor`
* `mxnet.nd.NDArray`
* `jax.numpy.DeviceArray`
* `paddle.Tensor`

#### Device Mapping for Tensors

To map tensor listener decoders to specific devices (CPUs/GPUs), add an argument to tensor data structures with direct GPU/TPU mapping to support re-mapping on mirrored node e.g.,

```
@PluginRegistrar.register
class MXNetTensor(Plugin):
    def __init__(self, load_mxnet_device=None, map_mxnet_devices=None, **kwargs):
```

where `map_mxnet_devices` should be `{'default': mxnet.gpu(0)` when `load_mxnet_device=mxnet.gpu(0)` and `map_mxnet_devices=None`.
For instance, when `load_mxnet_device=mxnet.gpu(0)` or `load_mxnet_device="cuda:0"`, `map_mxnet_devices` can be set manually as a dictionary representing the source device as key and the target device as value for non-default device maps. 

Suppose we have the following wrapyfied function:

```
@MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_native_exchange",
                                         carrier="tcp", should_wait=True, load_mxnet_device=mxnet.cpu(0), 
                                         map_mxnet_devices={"cuda:0": "cuda:1", 
                                                             mxnet.gpu(1): "cuda:0", 
                                                             "cuda:3": "cpu:0", 
                                                             mxnet.gpu(2):  mxnet.gpu(0)})
        def exchange_object(self):
            msg = input("Type your message: ")
            ret = {"message": msg,
                   "mx_ones": mxnet.nd.ones((2, 4)),
                   "mxnet_zeros_cuda1": mxnet.nd.zeros((2, 3), ctx=mxnet.gpu(1)),
                   "mxnet_zeros_cuda0": mxnet.nd.zeros((2, 3), ctx=mxnet.gpu(0)),
                   "mxnet_zeros_cuda2": mxnet.nd.zeros((2, 3), ctx=mxnet.gpu(2)),
                   "mxnet_zeros_cuda3": mxnet.nd.zeros((2, 3), ctx=mxnet.gpu(3))}
            return ret,
```

then the source and target gpus 1 & 0 would be flipped, gpu 3 would be placed on cpu 0, and gpu 2 would be placed on gpu 0. Defining `mxnet.gpu(1): mxnet.gpu(0)` and `cuda:1`: `cuda:2` in the same mapping should raise an error since the same device is mapped to two different targets.

The plugins supporting remapping are:

* `mxnet.nd.NDArray`
* `torch.Tensor`
* `paddle.Tensor`

## Environment Variables

Wrapyfi reserves specific environment variable names for the functionality of its internal components:

* `WRAPYFI_PLUGINS_PATH`: Path/s to [plugin](#plugins) extension directories 
* `WRAPYFI_DEFAULT_COMMUNICATOR` or `WRAPYFI_DEFAULT_MWARE` (`WRAPYFI_DEFAULT_MWARE` overrides `WRAPYFI_DEFAULT_COMMUNICATOR` when both are provided): Name of default [<Communicator>](#usage) when non is provided as the second argument to the Wrapyfi decorator. 
        
