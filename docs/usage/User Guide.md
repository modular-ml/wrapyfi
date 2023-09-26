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
`@MiddlewareCommunicator.register(<Data structure type>, <Communicator>, <Class name>, <Port name>)` is automatically registered by Wrapyfi. 

The `<Data structure type>` is the publisher/listener type for a given method's return. The supported data
types are listed [here](<User Guide/Plugins.md#data-structure-types>) section.

The `<Communicator>` defines the communication medium e.g.: `yarp`, `ros2`, `ros`, or `zeromq`. The default communicator is `zeromq` but can be replaced by setting the environment variables `WRAPYFI_DEFAULT_COMMUNICATOR` or `WRAPYFI_DEFAULT_MWARE` (`WRAPYFI_DEFAULT_MWARE` overrides `WRAPYFI_DEFAULT_COMMUNICATOR` when both are provided) to the middleware of choice e.g.: 
        
```
        export WRAPYFI_DEFAULT_COMMUNICATOR=yarp
```

The `<Class name>` serves no purpose in the current Wrapyfi version, but has been left for future support of module-level decoration, 
where the methods don't belong to a class, and must therefore have a unique identifier for declaration in the 
[configuration files](<User Guide/Configuration.md#configuration>).

The `<Port name>` is the name used for the connected port and is dependent on the middleware platform. The listener and publisher receive 
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
Methods with a single return should be followed by a comma e.g., return encapsulated a, . This explicitly casts the return as a tuple to avoid confusion with list returns as single return element/s
```

Each of the list's returns is encapsulated with its own publisher and listener, with the named arguments transmitted as 
a single dictionary within the list. Notice that `encapsulated_a` returns a list of length 3, therefore, the first decorator contains 
3 list configurations as well. Note that by using a single `NativeObject` as a `<Data structure type>`, the same 
can be achieved. However, the YARP implementation of the `NativeObject` utilizes `BufferedPortBottle` and serializes the 
object before transmission. The `NativeObject` may result in a greater overhead and should only be used when multiple nesting depths are 
required or the objects within a list are not within the [supported data structure types](<User Guide/Plugins.md#data-structure-types>).


