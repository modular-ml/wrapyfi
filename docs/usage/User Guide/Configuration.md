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


