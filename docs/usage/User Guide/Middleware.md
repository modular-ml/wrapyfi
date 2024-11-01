## Middleware

```{note}
The `<Data structure type>` is the object type for a given method's return. The supported data types are listed [here](<Plugins.md#data-structure-types>) section.
```

Wrapyfi natively supports a [number of middleware](#middleware). However, more middleware could be added by:
* Creating a derived class that inherits from the base classes `wrapyfi.connect.Listener`, `wrapyfi.connect.Publisher`, `wrapyfi.connect.Client`, or `wrapyfi.connect.Server` depending on the communication pattern to be supported
* Decorating the classes inside scripts residing within:
  * the `listeners` directory with `@Listeners.register(<Data structure type>, <Communicator>)` 
  * the `publishers` directory with `@Publishers.register(<Data structure type>, <Communicator>)`
  * the `clients` directory with `@Clients.register(<Data structure type>, <Communicator>)`
  * the `servers` directory with `@Servers.register(<Data structure type>, <Communicator>)`
* Appending the script path where the class is defined to the `WRAPYFI_MWARE_PATHS` environment variable
* Ensure that the middleware communication pattern scripts reside within directories named `listeners`, `publishers`, `clients`, or `servers` nested inside the `WRAPYFI_MWARE_PATH` and that the directory contains an `__init__.py` file

### Natively Supported Middleware
- [YARP](https://www.yarp.it/yarp_swig.html)
- [ROS](http://wiki.ros.org/rospy)
- [ROS 2](https://docs.ros2.org/foxy/api/rclpy/index.html)
- [ZeroMQ](http://zeromq.org/) [*beta feature*]: 
  * `should_wait` trigger introduced with event monitoring
  * Event monitoring currently cannot be disabled [![planned](https://custom-icon-badges.demolab.com/badge/planned%20for%20Wrapyfi%20v0.5-%23C2E0C6.svg?logo=hourglass&logoColor=white)](https://github.com/modular-ml/wrapyfi/issues/99 "planned link")
- [Websocket](https://socket.io/) *Only PUB/SUB* [*alpha support*]
- [Zenoh](https://zenoh.io/) *Only PUB/SUB* [*alpha support*]
- [MQTT](https://mqtt.org) *Only PUB/SUB* [*alpha support*]