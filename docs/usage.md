# Usage

To Wrapify your code:

```
from wrapify.connect.wrapper import MiddlewareCommunicator

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
the_class.activate_communication("encapsulated_func", mode="publish")
while True:
    the_class.encapsulating_func(...)
```

The primary component for facilitating communication is the `MiddlewareCommunicator`. To register the 
functions for a given class, it should inherit the `MiddlewareCommunicator`. Any function decorated with
`@MiddlewareCommunicator.register(<Data structure type>, <Class name>, <Port name>)`. 

The `<Data structure type>` is the publisher/listener type for a given function's return. The supported data
types are listed in the [publishers and listeners](#publishers-and-listeners) section.
 
The `<Class name>` serves no purpose in the current Wrapify version, but has been left for future support of module-level decoration, 
where the functions don't belong to a class, and must therefore have a unique identifier for declaration in the 
[configuration files](#configuration) 

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
3 list configurations as well.

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
on Wrapify. To overcome this limitation, use the `ConfigManager` e.g.:
```
from wrapify.config.manager import ConfigManager
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

The publishers and listeners of the same message type should have identical constructor signatures. The current Wrapify version supports
4 types of messages (Yarp compatible only):

* **Image**: Transmits and receives a `cv2` or `numpy` image using either `yarp.BufferedPortImageRgb` or `yarp.BufferedPortImageFloat`
* **AudioChunk**: Transmits and receives a `numpy` audio chunk using `yarp.BufferedPortImageFloat`, concurrently transmitting the sound properties using `yarp.BufferedPortSound`
* **NativeObject**: Transmits a `json` string supporting all native python objects using `yarp.BufferedPortBottle`
* **Matrix**: Transmits a `numpy` matrix of any size [*coming soon*]
* **Properties**: Transmits properties [*coming soon*]
