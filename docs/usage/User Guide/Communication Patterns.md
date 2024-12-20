## Communication Patterns

Wrapyfi supports the publisher-subscriber [(PUB/SUB)](#publishers-and-listeners-subscribers-pub-sub) pattern as well as the request-reply [(REQ/REP)](#servers-and-clients-req-rep) pattern. 
The PUB/SUB pattern assumes message arguments are passed from the publisher-calling script to the publishing method. 
The publisher executes the method and the subscriber (listener) merely triggers the method call, awaits the publisher to execute the method, and returns the publisher's method returns.
The REQ/REP pattern on the other hand assumes arguments from the client (requester) are sent to the server (responder or replier). Once the server receives the request, it passes the arguments
to its own method, executes it, and replies to the client back with its method returns.

Communication patterns in Wrapyfi are set by passing the configuration `mode` argument to `activate_communication` method as described in the [configuration documentation](<Configuration.md#communication-modes>).

```{warning}
in REQ/REP, the requester transmits all arguments passed to the method as a dictionary encoded as a string. This is not ideal for predefined services, where the service expects a certain object/message type. A better approach would include the option to pass a single item of a certain value and type [![planned](https://custom-icon-badges.demolab.com/badge/planned%20for%20Wrapyfi%20v0.5-%23C2E0C6.svg?logo=hourglass&logoColor=white)](https://github.com/modular-ml/wrapyfi/issues/99 "planned link")
```

### Publishers and Listeners/Subscribers (PUB/SUB)

The publishers and listeners of the same message type should have identical constructor signatures. The current Wrapyfi version supports
4 universal message types for all middleware. The extended types such as `ROSMessage` and `ROS2Message` are exclusive to the provided middleware.

#### YARP:

```{note}
YARP publishers remain [persistent](https://www.yarp.it/latest/persistent_connections.html#:~:text=When%20a%20connection%20is%20made%20between%20two%20YARP,made%20whenever%20possible.%20These%20are%20called%20%22persistent%20connections%22.). To disable persistence, pass the argument `persistent=False` to the `@MiddlewareCommunicator.register` decorator.
```

All messages are transmitted using the `yarp` Python bindings

* **Image**: Transmits and receives a `cv2` or `numpy` image using either `yarp.BufferedPortImageRgb` or `yarp.BufferedPortImageFloat`. 
             When JPG conversion is specified, it uses a `yarp.BufferedPortBottle` message carrying a JPEG encoded string instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk with the sound properties using `yarp.Port` transporting `yarp.Sound`
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays and [other formats](<Plugins.md#data-structure-types>) using `yarp.BufferedPortBottle`
* **Properties**: Transmits properties [![planned](https://custom-icon-badges.demolab.com/badge/planned%20for%20Wrapyfi%20v0.5-%23C2E0C6.svg?logo=hourglass&logoColor=white)](https://github.com/modular-ml/wrapyfi/issues/99 "planned link")

#### ROS:

```{warning}
ROS requires a custom message to handle audio. This message must be compiled first before using Wrapyfi with ROS Audio. 
Refer to [these instructions for compiling Wrapyfi ROS services and messages](https://github.com/modular-ml/wrapyfi_ros_interfaces/blob/master/README.md).
```

All messages are transmitted using the `rospy` Python bindings as topic messages

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`. When JPG conversion is specified, uses the `sensor_messages.msg.CompressedImage` message instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `wrapyfi_ros_interfaces.msg.ROSAudioMessage`
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`
* **Properties**: Transmits and receives parameters  to/from the parameter server using the methods `rospy.set_param` and `rospy.get_param` respectively
* **ROSMessage**: Transmits and receives a single [ROS message](http://wiki.ros.org/msg) per return decorator. Note that currently, only common ROS interface messages 
                  are supported and detected automatically. This means that messages defined in common interfaces such as [std_msgs](http://wiki.ros.org/std_msgs), 
                  [geometry_msgs](http://wiki.ros.org/geometry_msgs), and [sensor_msgs](http://wiki.ros.org/sensor_msgs) can be directly 
                  returned by the method do not need to be converted to native types

#### ROS 2: 

```{warning}
ROS 2 requires a custom message to handle audio. This message must be compiled first before using Wrapyfi with ROS 2 Audio. 
Refer to [these instructions for compiling Wrapyfi ROS 2 services and messages](https://github.com/modular-ml/wrapyfi_ros2_interfaces/blob/master/README.md).
```

All messages are transmitted using the `rclpy` Python bindings as topic messages

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`. When JPG conversion is specified, uses the `sensor_messages.msg.CompressedImage` message instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `wrapyfi_ros2_interfaces.msg.ROS2AudioMessage`
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`
* **Properties**: Transmits properties [![planned](https://custom-icon-badges.demolab.com/badge/planned%20for%20Wrapyfi%20v0.5-%23C2E0C6.svg?logo=hourglass&logoColor=white)](https://github.com/modular-ml/wrapyfi/issues/99 "planned link")
* **ROS2Message**: Transmits and receives a single [ROS 2 message](https://docs.ros.org/en/humble/Concepts/About-ROS-Interfaces.html) per return decorator

#### ZeroMQ:

```{note}
ZeroMQ exchanges in REQ/REP rely on a broker with a dedicated socket. By default, Wrapyfi will not spawn a new connection to the socket when multiple threads are created. For multi-threaded applications, this leads to race conditions. We avoid that by detecting whether a new instance of the socket is available in the thread's local storage. This multi-threading-friendly mode is enabled by passing `multi_threaded=True` to the `@MiddlewareCommunicator.register` decorator. This is only recommended when registering methods that are going to be multi-threaded.
```

All messages are transmitted using the `zmq` Python bindings. Transmission follows the [proxied XPUB/XSUB pattern](https://rfc.zeromq.org/spec/29/)

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct. Note that all `Image` types
                    are transmitted as multipart messages, where the first element is the topic name and the second element is the header (e.g., timestamp), 
                    and the third element is the image itself 
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays and [other formats](<Plugins.md#data-structure-types>) using 
                    `zmq context.socket(zmq.PUB).send_multipart` for publishing and `zmq context.socket(zmq.SUB).receive_multipart` for receiving messages.
                    The `zmq.PUB` socket is wrapped in a `zmq.proxy` to allow multiple subscribers to the same publisher. Note that all `NativeObject` types
                    are transmitted as multipart messages, where the first element is the topic name and the second element is the message itself (Except for `Image`)
* **Properties**: Transmits properties [![planned](https://custom-icon-badges.demolab.com/badge/planned%20for%20Wrapyfi%20v0.5-%23C2E0C6.svg?logo=hourglass&logoColor=white)](https://github.com/modular-ml/wrapyfi/issues/99 "planned link")

#### Websocket:

```{note}
Websocket assumes a server is running on the specified address and port. The forwarding of messages can only be done manually by the user. An example server can be found [here](https://github.com/fabawi/wrapyfi/tree/main/wrapyfi/examples/websockets/websocket_server.py) 
```

```{note}
Unlike the majority of middleware supported by Wrapyfi, websockets are bidirectional, meaning that the publisher can also be a listener. This allows Wrapyfi to support multiple publishers on the same topic ([namespaces](https://socket.io/docs/v4/namespaces/) and [rooms](https://socket.io/docs/v4/rooms/)) 
```

All messages are transmitted using the `python-socketio` Python bindings. Transmission follows the [socket.io protocol](https://socket.io/docs/v4/)

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays and [other formats](<Plugins.md#data-structure-types>) using 
                    `socketio.emit` for publishing and `socketio.on` for receiving messages
* **Properties**: Transmits properties [![planned](https://custom-icon-badges.demolab.com/badge/planned%20for%20Wrapyfi%20v0.5-%23C2E0C6.svg?logo=hourglass&logoColor=white)](https://github.com/modular-ml/wrapyfi/issues/99 "planned link")

#### Zenoh:

All messages are transmitted using the `eclipse-zenoh` Python bindings. Transmission follows the [zenoh protocol](https://zenoh.io/)

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct as `zenoh.Bytes`
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct as `zenoh.Bytes`
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays and [other formats](<Plugins.md#data-structure-types>) as `zenoh.Bytes` using 
                    `zenoh.session.key.put` for publishing and an asynchronus callback for receiving messages

#### MQTT:

```{note}
MQTT runs on a public online broker by default broker.emqx.io for convenience (no setup required), however, it is recommended to use a local broker like [Mosquitto](https://mosquitto.org/download/) for production.   
```

All messages are transmitted using the `paho-mqtt` Python bindings. Transmission follows the [MQTT protocol](https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html)

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays and [other formats](<Plugins.md#data-structure-types>) using 
                    `client.publish` for publishing and `client.on_message` for receiving messages
* **Properties**: Transmits properties [![planned](https://custom-icon-badges.demolab.com/badge/planned%20for%20Wrapyfi%20v0.5-%23C2E0C6.svg?logo=hourglass&logoColor=white)](https://github.com/modular-ml/wrapyfi/issues/99 "planned link")


### Servers and Clients (REQ/REP)

The servers and clients of the same message type should have identical constructor signatures. The current Wrapyfi version supports
3 universal message types for all middleware. The extended types such as `ROSMessage` and `ROS2Message` are exclusive to the provided middleware.

#### YARP:

All messages are transmitted using the `yarp` Python bindings [for RPC communication](https://www.yarp.it/latest/rpc_ports.html).
The requester encodes its arguments as a `json` string supporting all native Python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `yarp.Bottle`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image encoded as a `json` string using `yarp.Bottle`. *JPG conversion is currently not supported* 
* **AudioChunk**: Transmits and receives a `numpy` audio chunk encoded as a `json` string using `yarp.Bottle`
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `yarp.Bottle`

#### ROS:

```{warning}
ROS requires a custom service to handle audio. This service must be compiled first before using Wrapyfi with ROS Audio. 
Refer to [these instructions for compiling Wrapyfi ROS services and messages](https://github.com/modular-ml/wrapyfi_ros_interfaces/blob/master/README.md).
```

All messages are transmitted using the `rospy` Python bindings as services.
The requester encodes its arguments as a `json` string supporting all native Python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image` *JPG conversion is currently not supported* 
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `wrapyfi_ros_interfaces.msg.ROSAudioMessage`
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`

#### ROS 2:

```{warning}
ROS 2 requires custom services to handle arbitrary messages. These services must be compiled first before using Wrapyfi in this mode. 
Refer to [these instructions for compiling Wrapyfi ROS 2 services](https://github.com/modular-ml/wrapyfi_ros2_interfaces/blob/master/README.md).
```

All messages are transmitted using the rclpy Python bindings as services.
The requester encodes its arguments as a `json` string supporting all native Python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `wrapyfi_ros2_interfaces.msg.ROS2AudioMessage`
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`

#### ZeroMQ:

All messages are transmitted using the `zmq` Python bindings. Transmission follows the [proxied XREP/XREQ pattern](http://wiki.zeromq.org/tutorials:dealer-and-router)
The requester encodes its arguments as a `json` string supporting all native Python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `zmq context.socket(zmq.REQ).send_multipart`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct
* **NativeObject**: Transmits and receives a `json` string supporting all native Python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using 
                    `zmq context.socket(zmq.REP)` for replying and `zmq context.socket(zmq.REQ)` for receiving messages


### Publisher- and Listener-specific Arguments

```{warning}
Differences are expected between the returns of publishers and listeners, sometimes due to compression methods 
(e.g., setting `jpg=True` when transmitting an **Image** compresses the image but the encoding remains the same), 
intentional setting of different devices for different tensors (refer to [device mapping for tensors](<Plugins.md#device-mapping-for-tensors>)), 
and differences in library versions between receiving and transmitting plugins (refer to [plugins](<Plugins.md#plugins>)). 
```

To direct arguments specifically toward the publisher or subscriber without exposing one or the other to the same argument values, the corresponding arguments can be added to the dictionary `listener_kwargs` to control the listener only, or `publisher_kwargs` to control the publisher only. Both dictionaries can be passed directly to the Wrapyfi decorator.
Since the transmitting and receiving arguments should generally be the same regardless of the communication pattern, `publisher_kwargs` and `listener_kwargs` also apply to the servers and clients respectively.

