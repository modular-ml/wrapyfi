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



