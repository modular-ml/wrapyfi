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
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays and [other formats](<Plugins.md#data-structure-types>) using `yarp.BufferedPortBottle`
* **Properties**: Transmits properties [*coming soon*]

*(ROS)*:

All messages are transmitted using the rospy python bindings as topic messages

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`. When JPG conversion is specified, uses the `sensor_messages.msg.CompressedImage` message instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `sensor_messages.msg.Image`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`
* **Properties**: Transmits and receives parameters  to/from the parameter server using the methods `rospy.set_param` and `rospy.get_param` respectively
* **ROSMessage**: Transmits and receives a single [ROS message](http://wiki.ros.org/msg) per return decorator. Note that currently, only common ROS interface messages 
                  are supported and detected automatically. This means that messages defined in common interfaces such as [std_msgs](http://wiki.ros.org/std_msgs), 
                  [geometry_msgs](http://wiki.ros.org/geometry_msgs), and [sensor_msgs](http://wiki.ros.org/sensor_msgs) can be directly 
                  returned by the method do not need to be converted to native types

*(ROS2)*: 

```{warning}
ROS2 requires a custom message to deal with audio. This message must be compiled first before using Wrapyfi with ROS2 Audio. 
Refer to [these instructions for compiling Wrapyfi ROS2 services and messages](https://github.com/fabawi/wrapyfi/tree/master/wrapyfi_extensions/wrapyfi_ros2_interfaces/README.md).
```

All messages are transmitted using the rclpy python bindings as topic messages

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`. When JPG conversion is specified, uses the `sensor_messages.msg.CompressedImage` message instead
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `wrapyfi_ros2_interfaces.msg.ROS2AudioMessage`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`
* **Properties**: Transmits properties [*coming soon*]
* **ROS2Message**: Transmits and receives a single [ROS2 message](https://docs.ros.org/en/humble/Concepts/About-ROS-Interfaces.html) per return decorator [*coming soon*]

*(ZeroMQ)*:

All messages are transmitted using the zmq python bindings. Transmission follows the [proxied XPUB/XSUB pattern](https://rfc.zeromq.org/spec/29/)

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct. Note that all `Image` types
                    are transmitted as multipart messages, where the first element is the topic name and the second element is the header (e.g., timestamp), 
                    and the third element is the image itself 
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays and [other formats](<Plugins.md#data-structure-types>) using 
                    `zmq context.socket(zmq.PUB).send_multipart` for publishing and `zmq context.socket(zmq.SUB).receive_multipart` for receiving messages.
                    The `zmq.PUB` socket is wrapped in a `zmq.proxy` to allow multiple subscribers to the same publisher. Note that all `NativeObject` types
                    are transmitted as multipart messages, where the first element is the topic name and the second element is the message itself (Except for `Image`)
* **Properties**: Transmits properties [*coming soon*]


### Servers and Clients (REQ/REP)

The servers and clients of the same message type should have identical constructor signatures. The current Wrapyfi version supports
3 universal types of messages for all middleware. The extended types such as `ROSMessage` and `ROS2Message` are exclusive to the provided middleware.

*(YARP)*:

All messages are transmitted using the yarp python bindings [for RPC communication](https://www.yarp.it/latest/rpc_ports.html).
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `yarp.Bottle`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image encoded as a `json` string using `yarp.Bottle`. 
* **AudioChunk**: Transmits and receives a `numpy` audio chunk encoded as a `json` string using `yarp.Bottle` [*coming soon*]
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `yarp.Bottle`

*(ROS)*:

All messages are transmitted using the rospy python bindings as services.
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `sensor_messages.msg.Image`
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`

*(ROS2)*:

```{warning}
ROS2 requires custom services to deal with arbitrary messages. These services must be compiled first before using Wrapyfi in this mode. 
Refer to [these instructions for compiling Wrapyfi ROS2 services](../../ros2_interfaces_lnk.md).
```

All messages are transmitted using the rclpy python bindings as services.
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image using `sensor_messages.msg.Image`
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `sensor_messages.msg.Image` [*coming soon*]
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `std_msgs.msg.String`

*(ZeroMQ)*:

All messages are transmitted using the zmq python bindings. Transmission follows the [proxied XREP/XREQ pattern](http://wiki.zeromq.org/tutorials:dealer-and-router)
The requester encodes its arguments as a `json` string supporting all native python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using `zmq context.socket(zmq.REQ).send_multipart`.
The requester formats its arguments as *(\[args\], {kwargs})*

* **Image**: Transmits and receives a `cv2` or `numpy` image wrapped in the `NativeObject` construct
* **AudioChunk**: Transmits and receives a `numpy` audio chunk wrapped in the `NativeObject` construct [*coming soon*]
* **NativeObject**: Transmits and receives a `json` string supporting all native python objects, `numpy` arrays, and [other formats](<Plugins.md#data-structure-types>) using 
                    `zmq context.socket(zmq.REP)` for replying and `zmq context.socket(zmq.REQ)` for receiving messages


### Publisher- and Listener-specific Arguments

```{warning}
Differences are expected between the returns of publishers and listeners, sometimes due to compression methods (e.g., setting `jpg=True` when transmitting an **Image** compresses the image but the encoding remains the same), intentional setting of different devices for different tensors (refer to [device mapping for tensors](<Plugins.md#device-mapping-for-tensors>)), and differences in library versions between receiving and transmitting plugins (refer to [plugins](<Plugins.md#plugins>)). 
```

To direct arguments specifically toward the publisher or subscriber without exposing one or the other to the same argument values, the corresponding arguments can be added to the dictionary `listener_kwargs` to control the listener only, or `publisher_kwargs` to control the publisher only. Both dictionaries can be passed directly to the Wrapyfi decorator.
Since the transmitting and receiving arguments should generally be the same regardless of the communication pattern, `publisher_kwargs` and `listener_kwargs` also apply to the servers and clients respectively.

