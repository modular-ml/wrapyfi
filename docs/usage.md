# User Guide

To Wrapyfi your code:

```
from wrapyfi.connect.wrapper import MiddlewareCommunicator

class TheClass(MiddlewareCommunicator)
        ...
           
        @MiddlewareCommunicator.register(...)
        @MiddlewareCommunicator.register(...)
        def encapsulated_method(...):
            ...
            return encapsulated_a, encapsulated_b
        
        def encapsulating_method(...)
            ...
            encapsulated_a, encapsulated_b = self.encapsulated_method(...)
            ...
            return result,


the_class = TheClass()
the_class.activate_communication(the_class.encapsulated_method, mode="publish")
while True:
    the_class.encapsulating_method(...)
```

The primary component for facilitating communication is the `MiddlewareCommunicator`. To register the 
methods for a given class, it should inherit the `MiddlewareCommunicator`. Any method decorated with
`@MiddlewareCommunicator.register(<Data structure type>, <Communicator>, <Class name>, <Port name>)`. 

The `<Data structure type>` is the publisher/listener type for a given method's return. The supported data
types are listed in the [publishers and listeners](#publishers-and-listeners) section.

The `<Communicator>` defines the communication medium e.g.: `yarp`, `ros2`, `ros`, or `zeromq`. The default communicator is `zeromq` but can be replaced by setting the environment variables `WRAPYFI_DEFAULT_COMMUNICATOR` or `WRAPYFI_DEFAULT_MWARE` (`WRAPYFI_DEFAULT_MWARE` overrides `WRAPYFI_DEFAULT_COMMUNICATOR` when both are provided) to the middleware of choice e.g.: 
        
```
        export WRAPYFI_DEFAULT_COMMUNICATOR=yarp
```

The `<Class name>` serves no purpose in the current Wrapyfi version, but has been left for future support of module-level decoration, 
where the methods don't belong to a class, and must therefore have a unique identifier for declaration in the 
[configuration files](#configuration).

The `<Port name>` is the name used for the connected port and is dependent on the Middleware platform. The listener and publisher receive 
the same port name.

The `@MiddlewareCommunicator.register` decorator is defined for each of the method's returns in the 
same order. As shown in the example above, the first decorator defines the properties of `encapsulated_a`'s 
publisher and listener, whereas the second decorator belongs to `encapsulated_b`. A decorated method must always return a tuple which can easily
be enforced by adding a `comma` after the return in case a single variable is returned. Lists are also supported for 
single returns e.g.:
```
        @MiddlewareCommunicator.register([..., {...}], [..., {...}], [...])
        @MiddlewareCommunicator.register(...)
        def encapsulated_method(...):
            ...
            encapsulated_a = [[...], [...], [...]]
            ...
            return encapsulated_a, encapsulated_b
```

```{warning}
Methods with a single return should be followed by a comma e.g., return encapsulated a, . This explicitly casts the return as a tuple to avoid confusion with list returns as single return elements/
```

Each of the list's returns is encapsulated with its own publisher and listener, with the named arguments transmitted as 
a single dictionary within the list. Notice that `encapsulated_a` returns a list of length 3, therefore, the first decorator contains 
3 list configurations as well. Note that by using a single `NativeObject` as a `<Data structure type>`, the same 
can be achieved. However, the YARP implementation of the `NativeObject` utilizes `BufferedPortBottle` and serializes the 
object before transmission. The `NativeObject` may result in a greater overhead and should only be used when multiple nesting depths are 
required or the objects within a list are not within the [supported data structure types](#publishers-and-listeners).


## Configuration
The `MiddlewareCommunicator`'s child class method modes can be independently set to:

* **publish**: Run the method and publish the results using the middleware's transmission protocol
* **listen**: Skip the method and wait for the publisher with the same port name to transmit a message, eventually returning the received message

* **reply**: Run the method and publish the results using the middleware's transmission protocol. Arguments are received from the requester
* **request**: Send a request to the replier in the form of arguments passed to the method. Skip the method and wait for the replier with the same port name to transmit a message, eventually returning the received message

* **none**(default): Run the method as usual without triggering publish, listen, request or reply. *hint*: Setting the mode to `None` (or `null` within a yaml configuration file) has the same effect
* **disable**: Disables the method and returns None for all its returns. Caution should be taken when disabling a method since it 
could break subsequent calls

These properties can be set by calling: 

`activate_communication(<Method name>, mode=<Mode>)` 

for each decorated method within the class. This however requires modifying your scripts for each machine or process running
on Wrapyfi. To overcome this limitation, use the `ConfigManager` e.g.:
```
from wrapyfi.config.manager import ConfigManager
ConfigManager(<Configuration file path *.yml>)
``` 

The `ConfigManager` is a singleton that must be called once before the initialization of any `MiddlewareCommunicator`. Initializing it 
multiple times has no effect. This limitation was created by design to avoid loading the configuration file multiple times.

The `<Configuration file path *.yml>`'s configuration file has a very simple format e.g.:

```
TheClass:
  encapsulated_method: "publish"

```
Where `TheClass` is the class name, `encapsulated_method` is the method's name, and `publish` is the transmission mode.
This is useful when running the same script on multiple machines, where one is set to publish and the other listens. 
Multiple instances of the same class' method can have different modes, which can be set independently using the configuration file. This
can be achieved by providing the mode as a list:

```
TheClass:
  encapsulated_method: 
        "publish"
        null
        "listen"
        "listen"
        "disable"
        null
```

where the list element index corresponds to the instance index. When providing a list, the number of list elements should correspond to the number 
of instances. If the number of instances exceeds the list length, the script exits and raises an error.


## Communication Patterns

Wrapyfi supports the publisher-subscriber [(PUB/SUB)](#publishers-and-listenerssubscribers-pubsub) pattern as well as the request-reply (REQ/REP) pattern. 
The PUB/SUB pattern assumes message arguments are passed from the publisher-calling script to the publishing method. 
The publisher executes the method and the subscriber (listener) merely triggers the method call, awaits the publisher to execute the method, and returns the publisher's method returns.
The REQ/REP pattern on the other hand assumes arguments from the client (requester) are sent to the server (responder or replier). Once the server receives the request, it passes the arguments
to its own method, executes it, and replies to the client back with its method returns.


### Publishers and Listeners/Subscribers (PUB/SUB)

The publishers and listeners of the same message type should have identical constructor signatures. The current Wrapyfi version supports
4 universal types of messages for all middleware. The extended types such as `ROSMessage` and `ROS2Message` are exclusive to the provided middleware.

*(YARP)*:

All messages are transmitted using the yarp python bindings

* **Image**: Transmits and receives a `cv2` or `numpy` image using either `yarp.BufferedPortImageRgb` or `yarp.BufferedPortImageFloat`. 
             When JPG conversion is specified, use a `yarp.BufferedPortBottle` message carrying a JPEG encoded string instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `yarp.BufferedPortImageFloat`, concurrently transmitting the sound properties using `yarp.BufferedPortSound`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays and [other formats](#data-formats) using `yarp.BufferedPortBottle`
* **Properties**: Transmits properties [*coming soon*]

*(ROS)*:

All messages are transmitted using the rospy python bindings as topic messages

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`. When JPG conversion is specified, uses the `sensor_messages.msg.CompressedImage` message instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `sensor_messages.msg.Image`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `std_msgs.msg.String`
* **Properties**: Transmits and receives parameters  to/from the parameter server using the methods `rospy.set_param` and `rospy.get_param` respectively
* **ROSMessage**: Transmits and receives a single [ROS message](http://wiki.ros.org/msg) per return decorator. Note that currently, only common ROS interface messages 
                  are supported and detected automatically. This means that messages defined in common interfaces such as [std_msgs](http://wiki.ros.org/std_msgs), 
                  [geometry_msgs](http://wiki.ros.org/geometry_msgs), and [sensor_msgs](http://wiki.ros.org/sensor_msgs) can be directly 
                  returned by the method do not need to be converted to native types

*(ROS 2)*: 

All messages are transmitted using the rclpy python bindings as topic messages

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`. When JPG conversion is specified, uses the `sensor_messages.msg.CompressedImage` message instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `sensor_messages.msg.Image`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `std_msgs.msg.String`
* **Properties**: Transmits properties [*coming soon*]
* **ROS2Message**: Transmits and receives a single [ROS2 message](https://docs.ros.org/en/humble/Concepts/About-ROS-Interfaces.html) per return decorator [*coming soon*]

*(ZeroMQ)*:

All messages are transmitted using the zmq python bindings. Transmission follows the [proxied XPUB/XSUB pattern](https://rfc.zeromq.org/spec/29/)

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct. Note that all `Image` types
                    are transmitted as multipart messages, where the first element is the topic name and the second element is the header (e.g., timestamp), 
                    and the third element is the image itself 
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays and [other formats](#data-formats) using 
                    `zmq context.socket(zmq.PUB).send_multipart` for publishing and `zmq context.socket(zmq.SUB).receive_multipart` for receiving messages.
                    The `zmq.PUB` socket is wrapped in a `zmq.proxy` to allow multiple subscribers to the same publisher. Note that all `NativeObject` types
                    are transmitted as multipart messages, where the first element is the topic name and the second element is the message itself (Except for `Image`)
* **Properties**: Transmits properties [*coming soon*]


### Servers and Clients (REQ/REP)

The servers and clients of the same message type should have identical constructor signatures. The current Wrapyfi version supports
3 universal types of messages for all middleware. The extended types such as `ROSMessage` and `ROS2Message` are exclusive to the provided middleware.

*(YARP)*:

All messages are transmitted using the yarp python bindings [for RPC communication](https://www.yarp.it/latest/rpc_ports.html).
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `yarp.Bottle`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image encoded as a `json` string using `yarp.Bottle`. 
* **AudioChunk**: Transmits and receives a `numpy` audio chunk encoded as a `json` string using `yarp.Bottle` [*coming soon*]
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `yarp.Bottle`

*(ROS)*:

All messages are transmitted using the rospy python bindings as services.
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `std_msgs.msg.String`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `sensor_messages.msg.Image`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `std_msgs.msg.String`

*(ROS2)*:

```{warning}
ROS2 requires custom services to deal with arbitrary messages. These services must be compiled first before using Wrapyfi in this mode. 
Refer to [these instructions for compiling Wrapyfi ROS2 services](../wrapyfi_extensions/wrapyfi_ros2_interfaces/README.md).
```

All messages are transmitted using the rclpy python bindings as services.
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `std_msgs.msg.String`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image` [*coming soon*]
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `sensor_messages.msg.Image` [*coming soon*]
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `std_msgs.msg.String`

*(ZeroMQ)*:

All messages are transmitted using the zmq python bindings. Transmission follows the [proxied XREP/XREQ pattern](http://wiki.zeromq.org/tutorials:dealer-and-router)
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using `zmq context.socket(zmq.REQ).send_multipart`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct [*coming soon*]
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct [*coming soon*]
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-formats) using 
                    `zmq context.socket(zmq.REP)` for replying and `zmq context.socket(zmq.REQ)` for receiving messages


### Publisher- and Listener-specific Arguments

```{warning}
Differences are expected between the returns of publishers and listeners, sometimes due to compression methods (e.g., setting `jpg=True` when transmitting an **Image** compresses the image but the encoding remains the same), intentional setting of different devices for different tensors (refer to [device mappign for tensors](#device-mapping-for-tensors)), and differences in library versions between receiving and transmitting pugins (refer to plugins(#plugins). 
```

To direct arguments specifically toward the publisher or subscriber without exposing one or the other to the same argument values, the corresponding arguments can be added to the dictionary `listener_kwargs` to control the listener only, or `publisher_kwargs` to control the publisher only. Both dictionaries can be passed directly to the Wrapyfi decorator.
Since the transmitting and receiving arguments should generally be the same regardless of the communication pattern, `publisher_kwargs` and `listener_kwargs` also apply to the servers and clients respectively.

### Plugins

The **NativeObject** message type supports structures beyond native python objects. Wrapyfi already supports a number of non-native objects including numpy arrays and tensors. Wrapyfi can be extended to support objects by using the plugin API. All currently supported plugins by Wrapyfi can be found in the [plugins directory](../wrapyfi/plugins). Plugins can be added by:
* Creating a derived class that inherits from the base class `wrapify.utils.Plugin`
* Overriding the `encode` method for converting the object to a `json` serializable string. Deserializing the string is performed within the overridden `decode` method
* Specifying custom object properties by defining keyword arguments for the class constructor. These properties can be passed directly to the Wrapyfi decorator
* Decorating the class with `@PluginRegistrar.register` and appending the plugin to the list of supported objects
* Appending the script path where the class is defined to the `WRAPYFI_PLUGINS_PATH` environment variable
* Ensure that the plugin resides within a directory named `plugins` residing inside the `WRAPYFI_PLUGINS_PATH` and that the directory contains an `__init__.py` file

```{warning}
Due to differences in versions, the decoding may result in inconsitent outcomes, which must be handled for all versions e.g., MXNet plugin differences are handles in the existing plugin. 
```

#### Data Structure Types

Other than native python objects, the following objects are supported:

* `numpy.ndarray` and `numpy.generic`
* `pandas.DataFrame` and `pandas.Series`
* `torch.Tensor`
* `tensorflow.Tensor` and `tensorflow.EagerTensor`
* `mxnet.nd.NDArray`
* `jax.numpy.DeviceArray`
* `paddle.Tensor`
* `PIL.Image`

#### Device Mapping for Tensors

To map tensor listener decoders to specific devices (CPUs/GPUs), add an argument to tensor data structures with direct GPU/TPU mapping to support re-mapping on mirrored nodes e.g.,

```
@PluginRegistrar.register
class MXNetTensor(Plugin):
    def __init__(self, load_mxnet_device=None, map_mxnet_devices=None, **kwargs):
```

where `map_mxnet_devices` should be `{'default': mxnet.gpu(0)` when `load_mxnet_device=mxnet.gpu(0)` and `map_mxnet_devices=None`.
For instance, when `load_mxnet_device=mxnet.gpu(0)` or `load_mxnet_device="cuda:0"`, `map_mxnet_devices` can be set manually as a dictionary representing the source device as key and the target device as value for non-default device maps. 

Suppose we have the following Wrapyfied method:

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

#### Serialization

```{warning}
When encoding dictionaries, `json` supports string keys only and converts any instances of `int` keys to string, causing a difference between the publisher and subscriber returns. It is best to avoid using `int` keys, otherwise handle the difference on the receiving end.
```

Wrapyfi currently supports JSON as the only serializer. This introduces a number of limitations (beyond serializing native python objects only by default), including:

* dictionary keys cannot be integers. Integers are automatically converted to strings
* Tuples are converted to lists. Sets are not serializable. Tuples and sets are encoded as strings and restored on listening, which resolves this limitation but adds to the encoding overhead. This conversion is supported in Wrapyfi

## Environment Variables

Wrapyfi reserves specific environment variable names for the functionality of its internal components:

* `WRAPYFI_PLUGINS_PATH`: Path/s to [plugin](#plugins) extension directories 
* `WRAPYFI_DEFAULT_COMMUNICATOR` or `WRAPYFI_DEFAULT_MWARE` (`WRAPYFI_DEFAULT_MWARE` overrides `WRAPYFI_DEFAULT_COMMUNICATOR` when both are provided): Name of default [<Communicator>](#usage) when non is provided as the second argument to the Wrapyfi decorator. 

ZeroMQ requires socket configurations that can be passed as arguments to the respective middleware constructor (through the Wrapyfi decorator) or using environment variables. Note that these configurations are needed both by the proxy and the message publisher and listener. The downside to such an approach is that all messages share the same configs. This can be achieved by setting:
        
* `WRAPYFI_ZEROMQ_SOCKET_IP`: IP address of the socket. Defaults to "127.0.0.1"
* `WRAPYFI_ZEROMQ_SOCKET_PUB_PORT`: The publishing socket port. Defaults to 5555
* `WRAPYFI_ZEROMQ_SOCKET_SUB_PORT`: The sub-socket port (listening port for the broker). Defaults to 5556
* `WRAPYFI_ZEROMQ_START_PROXY_BROKER`: Spawn a new broker proxy without running the [standalone proxy broker](../wrapyfi/standalone/zeromq_proxy_broker.py). Defaults to "True"
* `WRAPYFI_ZEROMQ_PROXY_BROKER_SPAWN`: Either spawn broker as a "process" or "thread". Defaults to "process")
* `WRAPYFI_ZEROMQ_PARAM_POLL_INTERVAL`: Polling interval in milliseconds for the parameter server. Defaults to 1 (**currently not supported**)
* `WRAPYFI_ZEROMQ_PARAM_REQREP_PORT`: The parameter server request-reply port. Defaults to 5659 (**currently not supported**)
* `WRAPYFI_ZEROMQ_PARAM_PUB_PORT`: The parameter server pub-socket port. Defaults to 5655 (**currently not supported**)
* `WRAPYFI_ZEROMQ_PARAM_SUB_PORT`: The parameter server sub-socket port. Defaults to 5656 (**currently not supported**)

ROS and ROS2 queue sizes can be set by:

* `WRAPYFI_ROS_QUEUE_SIZE`: Size of the queue buffer. Defaults to 5
* `WRAPYFI_ROS2_QUEUE_SIZE`: Size of the queue buffer. Defaults to 5
