## Environment Variables

Wrapyfi reserves specific environment variable names for the functionality of its internal components:

* `WRAPYFI_PLUGINS_PATH`: Path/s to [plugin](<Plugins.md#plugins>) extension directories 
* `WRAPYFI_DEFAULT_COMMUNICATOR` or `WRAPYFI_DEFAULT_MWARE` (`WRAPYFI_DEFAULT_MWARE` overrides `WRAPYFI_DEFAULT_COMMUNICATOR` when both are provided): Name of default [<Communicator>](<../User Guide.md#usage>) when non is provided as the second argument to the Wrapyfi decorator. 

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
* `WRAPYFI_WEBSOCKET_MONITOR_LISTENER_SPAWN`: Either spawn the websocket monitor listener as a "process" or "thread". Defaults to "thread" which is the only supported option for now
* `WRAPYFI_MQTT_BROKER_ADDRESS`: The MQTT broker address. Defaults to "broker.emqx.io"
* `WRAPYFI_MQTT_BROKER_PORT`: The MQTT broker port. Defaults to 1883

ROS and ROS 2 queue sizes can be set by:

* `WRAPYFI_ROS_QUEUE_SIZE`: Size of the queue buffer. Defaults to 5
* `WRAPYFI_ROS2_QUEUE_SIZE`: Size of the queue buffer. Defaults to 5
