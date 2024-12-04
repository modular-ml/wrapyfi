## Environment Variables

Wrapyfi reserves specific environment variable names for the functionality of its internal components:

* `WRAPYFI_PLUGIN_PATHS`: Path/s to [plugin](<Plugins.md#plugins>) extension directories 
* `WRAPYFI_MWARE_PATHS`: Path/s to [middleware](<Middleware.md#middleware>) extension directories. These are simply middleware classes that are not part of the core library
* `WRAPYFI_DEFAULT_COMMUNICATOR` or `WRAPYFI_DEFAULT_MWARE` (`WRAPYFI_DEFAULT_MWARE` overrides `WRAPYFI_DEFAULT_COMMUNICATOR` when both are provided): Name of default [<Communicator>](<../User Guide.md#usage>) when none is provided as the second argument to the Wrapyfi decorator

ZeroMQ requires socket configurations that can be passed as arguments to the respective middleware constructor (through the Wrapyfi decorator) or using environment variables. Note that these configurations are needed both by the proxy and the message publisher and listener. 
The downside to such an approach is that all messages share the same configs. Since the proxy broker spawns once on first trigger (if enabled) as well as a singleton subscriber monitoring instance, using environment variables is the recommended approach to avoid unintended behavior. 
This can be achieved by setting:

* `WRAPYFI_ZEROMQ_SOCKET_IP`: IP address of the socket. Defaults to "127.0.0.1"
* `WRAPYFI_ZEROMQ_SOCKET_PUB_PORT`: The publishing socket port. Defaults to 5555
* `WRAPYFI_ZEROMQ_SOCKET_SUB_PORT`: The sub-socket port (listening port for the broker). Defaults to 5556
* `WRAPYFI_ZEROMQ_PUBSUB_MONITOR_TOPIC`: The topic name for the pub-sub monitor. Defaults to "ZEROMQ/CONNECTIONS"
* `WRAPYFI_ZEROMQ_PUBSUB_MONITOR_LISTENER_SPAWN`: Either spawn the pub-sub monitor listener as a "process" or "thread". Defaults to "process"
* `WRAPYFI_ZEROMQ_START_PROXY_BROKER`: Spawn a new broker proxy without running the [standalone proxy broker](../../../wrapyfi/standalone/zeromq_proxy_broker.py). Defaults to "True"
* `WRAPYFI_ZEROMQ_PROXY_BROKER_SPAWN`: Either spawn broker as a "process" or "thread". Defaults to "process")
* `WRAPYFI_ZEROMQ_PARAM_POLL_INTERVAL`: Polling interval in milliseconds for the parameter server. Defaults to 1 (**currently not supported**)
* `WRAPYFI_ZEROMQ_PARAM_REQREP_PORT`: The parameter server request-reply port. Defaults to 5659 (**currently not supported**)
* `WRAPYFI_ZEROMQ_PARAM_PUB_PORT`: The parameter server pub-socket port. Defaults to 5655 (**currently not supported**)
* `WRAPYFI_ZEROMQ_PARAM_SUB_PORT`: The parameter server sub-socket port. Defaults to 5656 (**currently not supported**)
* `WRAPYFI_WEBSOCKET_SOCKET_IP`: IP address of the socket. Defaults to "127.0.0.1"
* `WRAPYFI_WEBSOCKET_SOCKET_PORT`: The socket port. Defaults to 5000
* `WRAPYFI_WEBSOCKET_NAMESPACE`: The socket namespace. Defaults to "/"
* `WRAPYFI_ZENOH_IP`: IP address of the Zenoh socket. Defaults to "127.0.0.1"
* `WRAPYFI_ZENOH_PORT`: The Zenoh socket port. Defaults to 7447
* `WRAPYFI_ZENOH_MODE`: The Zenoh mode indicating whether to use the router as a broker or adopt peer-to-peer communication. Defaults to "peer"
* `WRAPYFI_ZENOH_CONNECT`: The Zenoh connect endpoints seperated by a comma e.g., "tcp/127.0.0.1:7447,udp/127.0.0.1:7448". This overrides `WRAPYFI_ZENOH_IP` and `WRAPYFI_ZENOH_PORT`. Defaults to an empty list
* `WRAPYFI_ZENOH_LISTEN`: The Zenoh listen endpoints seperated by a comma e.g., "tcp/127.0.0.1:7446". Defaults to an empty list
* `WRAPYFI_ZENOH_CONFIG_FILEPATH`: The Zenoh configuration file path. Defaults to None. Conflicting keys are overriden by `WRAPYFI_ZENOH_IP`, `WRAPYFI_ZENOH_PORT`, `WRAPYFI_ZENOH_CONNECT`, and `WRAPYFI_ZENOH_LISTEN`
* `WRAPYFI_MQTT_BROKER_ADDRESS`: The MQTT broker address. Defaults to "broker.emqx.io"
* `WRAPYFI_MQTT_BROKER_PORT`: The MQTT broker port. Defaults to 1883

ROS and ROS 2 queue sizes can be set by:

* `WRAPYFI_ROS_QUEUE_SIZE`: Size of the queue buffer. Defaults to 5
* `WRAPYFI_ROS2_QUEUE_SIZE`: Size of the queue buffer. Defaults to 5
