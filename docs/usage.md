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
methods for a given class, the class should inherit the `MiddlewareCommunicator`. Any method decorated with
`@MiddlewareCommunicator.register(<Data structure type>, <Communicator>, <Class name>, <Topic name>)` is automatically registered by Wrapyfi. 

The `<Data structure type>` is the publisher/listener type for a given method's return. The supported data
types are listed [here](#data-structure-types) section.

The `<Communicator>` defines the communication medium e.g.: `yarp`, `ros2`, `ros`, or `zeromq`. The default communicator is `zeromq` but can be replaced by setting the environment variables `WRAPYFI_DEFAULT_COMMUNICATOR` or `WRAPYFI_DEFAULT_MWARE` (`WRAPYFI_DEFAULT_MWARE` overrides `WRAPYFI_DEFAULT_COMMUNICATOR` when both are provided) to the middleware of choice e.g.: 
        
```
        export WRAPYFI_DEFAULT_COMMUNICATOR=yarp
```

The `<Class name>` serves no purpose in the current Wrapyfi version, but has been left for future support of module-level decoration, 
where the methods don't belong to a class, and must therefore have a unique identifier for declaration in the 
[configuration files](#configuration).

The `<Topic name>` is the name used for the connected topic and is dependent on the middleware platform. The listener and publisher receive 
the same topic name.

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
Methods with a single return should be followed by a comma e.g., return encapsulated a, . This explicitly casts the return as a tuple to avoid confusion with list returns as single return element/s
```

Each of the list's returns is encapsulated with its own publisher and listener, with the named arguments transmitted as 
a single dictionary within the list. Notice that `encapsulated_a` returns a list of length 3, therefore, the first decorator contains 
3 list configurations as well. This is useful especially when transmitting multiple images or audio chunks over YARP, ROS, and ROS2.
Note that by using a single `NativeObject` as a `<Data structure type>`, the same 
can be achieved. However, the implementation of the `NativeObject` for most middleware serializes the 
objects as strings before transmission. The `NativeObject` may result in a greater overhead and should only be used when multiple nesting depths are 
required or the objects within a list are not within the [supported data structure types](#data-structure-types).

### Argument Passing
The `$` symbol is used in Wrapyfi to specify that a decorator should update its arguments according to the arguments of the decorated method. This can be useful when the decorator needs to modify its behavior during runtime. For instance:
```
...
        @MiddlewareCommunicator.register('NativeObject', 
           '$0', 'ExampleCls', '/example/example_arg_pass', 
           carrier='tcp', should_wait='$blocking')
          def example_arg_pass(self, mware, msg='', blocking=True):
```

Setting the decorator's keyword argument `should_wait='$blocking'` expects the decorated method to receive a boolean `blocking` argument, altering the encapsulating decorator's behavior when the encapsulated method is called. Setting the decorator's second argument to `$0` acquires the value of `mware` (the first argument passed to `example_arg_pass`) and sets it as the middleware for that method. These arguments take effect on the first invocation of a method. Changing arguments after the first invocation results in no change in behavior, unless a `MiddlewareCommunicator` inheriting class for a given method is [closed](#closing-and-deleting-classes). 

### Closing and Deleting Classes
Currently, closing a connection requires closing all connections established by every method within that class. 

```{warning}
Selectively deactivating method connections is planned for Wrapyfi v0.5.0.
```

To close and delete a `MiddlewareCommunicator` inheriting class means that the middleware connection will be disconnected gracefully. The class references will be removed from all registries, the communication ports will be freed, and the instance will be destroyed. To close a class instance:

```
# assuming an existing instance-> example_instance = ExampleCls()
example_instance.close()
del example_instance
```

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

where `<Method name>` is the method's name (`string` name of method by definition) and `<Mode>` is the transmission mode 
(`"publish"`, `"listen"`, `"reply"`, `request`, `"none"` | `None`, `"disable"`) depending on the communication pattern . 
The `activate_communication` method can be called multiple times. `<Method name>` could also be a class instance method, by calling:

`activate_communication(<MiddlwareCommunicator instance>.method_of_class_instance, mode=<Mode>)`

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

where `TheClass` is the class name, `encapsulated_method` is the method's name, and `publish` is the transmission mode.
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

Wrapyfi supports the publisher-subscriber [(PUB/SUB)](#publishers-and-listeners-subscribers-pub-sub) pattern as well as the request-reply [(REQ/REP)](#servers-and-clients-req-rep) pattern. 
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
             When JPG conversion is specified, it uses a `yarp.BufferedPortBottle` message carrying a JPEG encoded string instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk with the sound properties using `yarp.Port` transporting `yarp.Sound`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays and [other formats](#data-structure-types) using `yarp.BufferedPortBottle`
* **Properties**: Transmits properties [*planned for Wrapyfi v0.5*]

*(ROS)*:

```{warning}
ROS requires a custom message to handle audio. This message must be compiled first before using Wrapyfi with ROS Audio. 
Refer to [these instructions for compiling Wrapyfi ROS services and messages](https://github.com/fabawi/wrapyfi/tree/master/wrapyfi_extensions/wrapyfi_ros_interfaces/README.md).
```

All messages are transmitted using the rospy python bindings as topic messages

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`. When JPG conversion is specified, uses the `sensor_messages.msg.CompressedImage` message instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `wrapyfi_ros_interfaces.msg.ROSAudioMessage`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-structure-types) using `std_msgs.msg.String`
* **Properties**: Transmits and receives parameters  to/from the parameter server using the methods `rospy.set_param` and `rospy.get_param` respectively
* **ROSMessage**: Transmits and receives a single [ROS message](http://wiki.ros.org/msg) per return decorator. Note that currently, only common ROS interface messages 
                  are supported and detected automatically. This means that messages defined in common interfaces such as [std_msgs](http://wiki.ros.org/std_msgs), 
                  [geometry_msgs](http://wiki.ros.org/geometry_msgs), and [sensor_msgs](http://wiki.ros.org/sensor_msgs) can be directly 
                  returned by the method do not need to be converted to native types

*(ROS2)*: 

```{warning}
ROS 2 requires a custom message to handle audio. This message must be compiled first before using Wrapyfi with ROS 2 Audio. 
Refer to [these instructions for compiling Wrapyfi ROS 2 services and messages](https://github.com/fabawi/wrapyfi/tree/master/wrapyfi_extensions/wrapyfi_ros2_interfaces/README.md).
```

All messages are transmitted using the rclpy python bindings as topic messages

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`. When JPG conversion is specified, uses the `sensor_messages.msg.CompressedImage` message instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `wrapyfi_ros2_interfaces.msg.ROS2AudioMessage`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-structure-types) using `std_msgs.msg.String`
* **Properties**: Transmits properties [*planned for Wrapyfi v0.5*]
* **ROS2Message**: Transmits and receives a single [ROS 2 message](https://docs.ros.org/en/humble/Concepts/About-ROS-Interfaces.html) per return decorator

*(ZeroMQ)*:

All messages are transmitted using the zmq python bindings. Transmission follows the [proxied XPUB/XSUB pattern](https://rfc.zeromq.org/spec/29/)

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct. Note that all `Image` types
                    are transmitted as multipart messages, where the first element is the topic name and the second element is the header (e.g., timestamp), 
                    and the third element is the image itself 
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays and [other formats](#data-structure-types) using 
                    `zmq context.socket(zmq.PUB).send_multipart` for publishing and `zmq context.socket(zmq.SUB).receive_multipart` for receiving messages.
                    The `zmq.PUB` socket is wrapped in a `zmq.proxy` to allow multiple subscribers to the same publisher. Note that all `NativeObject` types
                    are transmitted as multipart messages, where the first element is the topic name and the second element is the message itself (Except for `Image`)
* **Properties**: Transmits properties [*planned for Wrapyfi v0.5*]


### Servers and Clients (REQ/REP)

The servers and clients of the same message type should have identical constructor signatures. The current Wrapyfi version supports
3 universal types of messages for all middleware. The extended types such as `ROSMessage` and `ROS2Message` are exclusive to the provided middleware.

*(YARP)*:

All messages are transmitted using the yarp python bindings [for RPC communication](https://www.yarp.it/latest/rpc_ports.html).
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-structure-types) using `yarp.Bottle`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image encoded as a `json` string using `yarp.Bottle`. *JPG conversion is currently not supported* 
* **AudioChunk**: Transmits and receives a `numpy` audio chunk encoded as a `json` string using `yarp.Bottle`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-structure-types) using `yarp.Bottle`

*(ROS)*:

```{warning}
ROS requires a custom service to handle audio. This service must be compiled first before using Wrapyfi with ROS Audio. 
Refer to [these instructions for compiling Wrapyfi ROS services and messages](https://github.com/fabawi/wrapyfi/tree/master/wrapyfi_extensions/wrapyfi_ros_interfaces/README.md).
```

All messages are transmitted using the rospy python bindings as services.
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-structure-types) using `std_msgs.msg.String`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image` *JPG conversion is currently not supported* 
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `wrapyfi_ros_interfaces.msg.ROSAudioMessage`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-structure-types) using `std_msgs.msg.String`

*(ROS2)*:

```{warning}
ROS 2 requires custom services to handle arbitrary messages. These services must be compiled first before using Wrapyfi in this mode. 
Refer to [these instructions for compiling Wrapyfi ROS 2 services](https://github.com/fabawi/wrapyfi/tree/master/wrapyfi_extensions/wrapyfi_ros2_interfaces/README.md).
```

All messages are transmitted using the rclpy python bindings as services.
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-structure-types) using `std_msgs.msg.String`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `wrapyfi_ros2_interfaces.msg.ROS2AudioMessage`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-structure-types) using `std_msgs.msg.String`

*(ZeroMQ)*:

All messages are transmitted using the zmq python bindings. Transmission follows the [proxied XREP/XREQ pattern](http://wiki.zeromq.org/tutorials:dealer-and-router)
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-structure-types) using `zmq context.socket(zmq.REQ).send_multipart`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](#data-structure-types) using 
                    `zmq context.socket(zmq.REP)` for replying and `zmq context.socket(zmq.REQ)` for receiving messages


### Publisher- and Listener-specific Arguments

```{warning}
Differences are expected between the returns of publishers and listeners, sometimes due to compression methods (e.g., setting `jpg=True` when transmitting an **Image** compresses the image but the encoding remains the same), intentional setting of different devices for different tensors (refer to [device mapping for tensors](#device-mapping-for-tensors)), and differences in library versions between receiving and transmitting plugins (refer to [plugins](#plugins)). 
```

To direct arguments specifically toward the publisher or subscriber without exposing one or the other to the same argument values, the corresponding arguments can be added to the dictionary `listener_kwargs` to control the listener only, or `publisher_kwargs` to control the publisher only. Both dictionaries can be passed directly to the Wrapyfi decorator.
Since the transmitting and receiving arguments should generally be the same regardless of the communication pattern, `publisher_kwargs` and `listener_kwargs` also apply to the servers and clients respectively.

## Communication Schemes

We introduce three communication schemes: **Mirroring**, **Channeling**, and **Forwarding**. 
These schemes are communication forms that can be useful in different scenarios.


### Mirroring

For the REQ/REP pattern, mirroring is a communication scheme that allows a client to send arguments to a server, 
and receive the method returns back from the server. As for the PUB/SUB pattern, mirroring allows a publisher to
send the returns of a method to a subscriber based on the publisher's method arguments. Following both patterns, 
the returns of a method are mirrored on the receiver and the sender side. This is useful when the pipeline for each
receiver is identical, but we would like to delegate the processing to different publishers when processing requires 
more resources than a single publisher can provide. 

#### Mirroring Example

In the [mirroring_example.py](https://github.com/fabawi/wrapyfi/blob/master/examples/communication_schemes/mirroring_example.py), 
the module transmits a user input message from the publisher to a listener (PUB/SUB pattern), and displays the message along with other native 
objects on the listener and publisher side. Similarly, we transmit a user input message from the server to a client (REQ/REP pattern),
when the client requests the message from the server. The example can be run from the 
[examples/communication_schemes/](https://github.com/fabawi/wrapyfi/blob/master/examples/communication_schemes) directory.


### Forwarding

Forwarding is a communication scheme that allows a server or publisher to forward the method arguments to another server or publisher (acting as a client or listener),
and in return, forwards the received messages to another client or listener. This is useful when the server or publisher is not able to communicate with the client or listener
directly due to limited middleware support on the client or listener side. The middle server or publisher can then act as a bridge between the two, and forward the messages
between them, effectively chaining the communication. The chain can be extended and is not limited to two servers or publishers.

#### Forwarding Example

In the [forwarding_example.py](https://github.com/fabawi/wrapyfi/blob/master/examples/communication_schemes/forwarding_example.py),
the module constantly publishes a string from `chain_A` to a listener on `chain_A`. The `chain_A` listener then forwards the message by publishing to `chain_B`. 
The string is then forwarded to a third instances which listens exclusively to `chain_B`, without needing to support the middleware used by `chain_A`.
The example can be run from the [examples/communication_schemes/](https://github.com/fabawi/wrapyfi/blob/master/examples/communication_schemes) directory.


### Channeling

Channeling differs from mirroring, in that there are multiple returns from a method. Disabling one or more of these returns
is possible, allowing the server or publisher to transmit the message to multiple channels, each with a different topic, and
potentially, a different middleware. This is useful for transmitting messages using the same method, but to different receivers
based on what they choose to receive. Not all clients or subscribers require all the messages from a method, and can therefore
selectively filter out what is needed and operate on that partial return.

#### Channeling Example

In the [channeling_example.py](https://github.com/fabawi/wrapyfi/blob/master/examples/communication_schemes/channeling_example.py),
the module constantly publishes three data types (**NativeObject**, **Image**, and **AudioChunk**) over one or more middlware. The listeners 
can then choose to receive one or more of these data types, depending on the middleware they support. When `--mware_...` for one of the
channels is not provided, it automatically disables the topic for that channel/s and returns a `None` type value. 
The example can be run from the [examples/communication_schemes/](https://github.com/fabawi/wrapyfi/blob/master/examples/communication_schemes) directory.


## Plugins

The **NativeObject** message type supports structures beyond native python objects. Wrapyfi already supports a number of non-native objects including numpy arrays and tensors. Wrapyfi can be extended to support objects by using the plugin API. All currently supported plugins by Wrapyfi can be found in the [plugins directory](https://github.com/fabawi/wrapyfi/tree/master/wrapyfi/plugins). Plugins can be added by:
* Creating a derived class that inherits from the base class `wrapyfi.utils.Plugin`
* Overriding the `encode` method for converting the object to a `json` serializable string. Deserializing the string is performed within the overridden `decode` method
* Specifying custom object properties by defining keyword arguments for the class constructor. These properties can be passed directly to the Wrapyfi decorator
* Decorating the class with `@PluginRegistrar.register` and appending the plugin to the list of supported objects
* Appending the script path where the class is defined to the `WRAPYFI_PLUGINS_PATH` environment variable
* Ensure that the plugin resides within a directory named `plugins` nested inside the `WRAPYFI_PLUGINS_PATH` and that the directory contains an `__init__.py` file

#### Plugin Example

An example for adding a plugin for a custom [Astropy](https://www.astropy.org/) object is provided in the [astropy_example.py example](https://github.com/fabawi/wrapyfi/blob/master/examples/encoders/astropy_example.py).
In the example, we append the example's directory to the `WRAPYFI_PLUGINS_PATH` environment variable and import the plugin. 
The plugin ([astropy_tables.py](https://github.com/fabawi/wrapyfi/blob/master/examples/encoders/plugins/astropy_tables.py)) in the [plugins](https://github.com/fabawi/wrapyfi/blob/master/examples/encoders/plugins) directory
is then used to encode and decode the custom object (from within the `examples/encoders/` directory): 

```
# create the publisher with default middleware (changed with --mware). The plugin is automatically loaded
python3 astropy_example.py --mode publish
# create the listener with default middleware (changed with --mware). The plugin is automatically loaded
python3 astropy_example.py --mode listen
```

from the two terminal outputs, the same object should be printed after typing a random message and pressing enter:

```
Method result: [{'message': 'hello world', 'astropy_table': <Table length=3>
  name     flux 
 bytes8  float64
-------- -------
source 1     1.2
source 2     2.2
source 3     3.1, 'list': [1, 2, 3]}, 'string', 0.4344, {'other': (1, 2, 3, 4.32)}]
```

```{warning}
Due to differences in versions, the decoding may result in inconsitent outcomes, which must be handled for all versions e.g., MXNet plugin differences are handled in the existing plugin. 
```

### Data Structure Types

Other than native python objects, the following objects are supported:

* `numpy.ndarray` and `numpy.generic`
* `pandas.DataFrame` and `pandas.Series` (pandas v1)
* `torch.Tensor`
* `tensorflow.Tensor` and `tensorflow.EagerTensor`
* `mxnet.nd.NDArray`
* `jax.numpy.DeviceArray`
* `paddle.Tensor`
* `PIL.Image`
* `pyarrow.StructArray`
* `xarray.DataArray` and `xarray.Dataset`
* `dask.array.Array` and `dask.dataframe.DataFrame`
* `zarr.core.Array` and `zarr.core.Group`
* `pint.Quantity`


### Device Mapping for Tensors

To map tensor listener decoders to specific devices (CPUs/GPUs), add an argument to tensor data structures with direct GPU/TPU mapping to support re-mapping on mirrored nodes e.g.,

```
@PluginRegistrar.register
class MXNetTensor(Plugin):
    def __init__(self, load_mxnet_device=None, map_mxnet_devices=None, **kwargs):
```

where `map_mxnet_devices` should be `{'default': mxnet.gpu(0)}` when `load_mxnet_device=mxnet.gpu(0)` and `map_mxnet_devices=None`.
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

### Serialization

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

ZeroMQ requires socket configurations that can be passed as arguments to the respective middleware constructor (through the Wrapyfi decorator) or using environment variables. Note that these configurations are needed both by the proxy and the message publisher and listener. 
The downside to such an approach is that all messages share the same configs. Since the proxy broker spawns once on first trigger (if enabled) as well as a singleton subscriber monitoring instance, using environment variables is the recommended approach to avoid unintended behavior. 
This can be achieved by setting:
        
* `WRAPYFI_ZEROMQ_SOCKET_IP`: IP address of the socket. Defaults to "127.0.0.1"
* `WRAPYFI_ZEROMQ_SOCKET_PUB_PORT`: The publishing socket port. Defaults to 5555
* `WRAPYFI_ZEROMQ_SOCKET_SUB_PORT`: The sub-socket port (listening port for the broker). Defaults to 5556
* `WRAPYFI_ZEROMQ_PUBSUB_MONITOR_TOPIC`: The topic name for the pub-sub monitor. Defaults to "ZEROMQ/CONNECTIONS"
* `WRAPYFI_ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN`: Either spawn the pub-sub monitor listener as a "process" or "thread". Defaults to "process"
* `WRAPYFI_ZEROMQ_START_PROXY_BROKER`: Spawn a new broker proxy without running the [standalone proxy broker](https://github.com/fabawi/wrapyfi/tree/master/wrapyfi/standalone/zeromq_proxy_broker.py). Defaults to "True"
* `WRAPYFI_ZEROMQ_PROXY_BROKER_SPAWN`: Either spawn broker as a "process" or "thread". Defaults to "process")
* `WRAPYFI_ZEROMQ_PARAM_POLL_INTERVAL`: Polling interval in milliseconds for the parameter server. Defaults to 1 (**currently not supported**)
* `WRAPYFI_ZEROMQ_PARAM_REQREP_PORT`: The parameter server request-reply port. Defaults to 5659 (**currently not supported**)
* `WRAPYFI_ZEROMQ_PARAM_PUB_PORT`: The parameter server pub-socket port. Defaults to 5655 (**currently not supported**)
* `WRAPYFI_ZEROMQ_PARAM_SUB_PORT`: The parameter server sub-socket port. Defaults to 5656 (**currently not supported**)

ROS and ROS 2 queue sizes can be set by:

* `WRAPYFI_ROS_QUEUE_SIZE`: Size of the queue buffer. Defaults to 5
* `WRAPYFI_ROS2_QUEUE_SIZE`: Size of the queue buffer. Defaults to 5
