## Configuration

In order to establish communication using any [communication pattern](<Communication%20Patterns.md#communication-patterns>) supported by Wrapyfi, the methods need to be activated by setting them to behave according to a compatible communication mode

### Communication Modes

Modes are configurations that define the behavior of a method, i.e., *should the method be executed?*, *should the method await a message from another method?*, *should the method publish its message?*, or *should the method await an acknowledgement from a message requester?*

Selecting the mode for each method is tha starting point for establishing communication through Wrapyfi. Wrapyfi supports
the common PUB/SUB and REQ/REP communication patterns, with different modes supported for each.
The `MiddlewareCommunicator`'s child class method modes can be independently set to accomodate the communication pattern. We separate the modes into their corresponding patterns:

#### Mode-Agnostic

* **none**(default): Run the method as usual without triggering `publish, listen, receive, reemit, transceive, request` or `reply`. *hint*: Setting the mode to `None` (or `null` within a yaml configuration file) has the same effect
* **disable**: Disables the method and returns None for all its returns. Caution should be taken when disabling a method since it 
could break subsequent calls

#### Modes for Publishers and Listeners/Subscribers (PUB/SUB)

* **publish**: Run the method and publish the results using the middleware's transmission protocol. The results of executing the method are returned directly from the method: The method runs as expected, with the additional publishing of its returns
* **listen**: Skip the method and wait for the publisher with the same port name to transmit a message, eventually returning the received message

* **receive**: Similar in behavior to the `listen` mode, but receives the input as an argument and executes the method assuming the argument/s are received from a publisher. The method is then run based on the received argument/s value/s and the returns are resulting from running the method itself, rather than passively listening for input only

* **reemit**: Similar in behavior to the `receive` mode, but also publishes the returns eventually, similar to the `publish` mode
* **transceive**: Similar in behavior to the `publish` mode, however, unlike publishing methods, it listens for returns from another publisher, and returns the resulting message instead, similar to the `listen` mode

The possible mode combinations for each method to communicate successfully with its corresponding methods operating on the same topic are:

* `publish` **<->** `listen`: Both method returns are identical
* `publish` **<->** `disable`: Method returns are not identical: This only works when `publish` decorater argument `should_wait=False`, since there is no listener, given `disable` mode always returns `None` and does not establish a connection with the publisher
* `publish` **<->** `none`: Method returns are not identical: This only works when `publish` decorater argument `should_wait=False`, since there is no listener, given `"none"`/`None` mode always runs the method and acquires its returns but does not establish a connection with the publisher
* `publish` **<->** `receive`: Method returns are not identical
* `transceive` **<->** `reemit`: Both method returns *could* be identical



#### Modes for Servers and Clients (REQ/REP)

* **reply**: Run the method and publish the results using the middleware's transmission protocol. Arguments are received from the requester
* **request**: Send a request to the replier in the form of arguments passed to the method. Skip the method and wait for the replier with the same port name to transmit a message, eventually returning the received message

The possible mode combinations for each method to communicate successfully with its corresponding methods operating on the same topic are:

* `request` **<->** `reply`: Both method returns are identical. For every request (client) there should be a corresponding reply (server) and would not operate (hangs indefinitely) without acknowledgement from the server

### Activating Communication Modes

The mode of each method can be set by calling: 

`activate_communication(<Method name>, mode=<Mode>)`

where `<Method name>` is the method's name (`string` name of method by definition) and `<Mode>` is the transmission mode 
(`"publish"`, `"listen"`,`"receive"`, `"transceive"`, `reemit`, `"reply"`, `request`, `"none"` | `None`, `"disable"`) depending on the communication pattern . 
The `activate_communication` method can be called multiple times. `<Method name>` could also be a class instance method, by calling:

`activate_communication(<MiddlwareCommunicator instance>.method_of_class_instance, mode=<Mode>)`

### Configuration Files

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


