
<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/4982924/194550571-e7738a6d-da05-4e0d-8904-8edf784ceef4.png">
</p>

# 

Wrapyfi is a middleware communication wrapper for transmitting data across nodes, without the need to
alter the operation pipeline of your python scripts. Wrapyfi introduces
a number of helper functions to make middleware integration possible without the need to learn an entire framework, just to parallelize your processes on 
multiple machines. 
Wrapyfi supports [YARP](https://www.yarp.it/yarp_swig.html), [ROS](http://wiki.ros.org/rospy), [ROS 2](https://docs.ros2.org/foxy/api/rclpy/index.html), and [ZeroMQ](http://zeromq.org/).

To Wrapyfi a class, simply add the decorators describing the publisher and listener parameters. Wrapyfi imposes an object-oriented
requirement on your coding style: All wrapyfi compatible functions need to be defined within a class. 

## Installation

Before using Wrapyfi, YARP, ROS, or ZeroMQ must be installed. 

Follow the [YARP installation guide](https://github.com/modular-ml/wrapyfi-interfaces/blob/main/wrapyfi_interfaces/robots/icub_head/README.md#installing-yarp).
Note that the iCub package is not needed for Wrapyfi to work and does not have to be installed if you do not intend on using the iCub robot.

For installing ROS, follow the [ROS installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu). 
We recommend installing ROS on conda using the [RoboStack](https://github.com/RoboStack/ros-noetic) environment.

For installing ROS 2, follow the [ROS 2 installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html). 
We recommend installing ROS 2 on conda using the [RoboStack](https://github.com/RoboStack/ros-humble) environment.

ZeroMQ can be installed using pip: `pip install pyzmq`. 
The xpub-xsub pattern followed in our ZeroMQ implementation requires a proxy broker. A broker is spawned by default as a daemon process.
To avoid automatic spawning, pass the argument `start_proxy_broker=False` to the method register decorator. 
A standalone broker can be found [here](wrapyfi/standalone/zmq_proxy_broker.py)

#### compatibility
* Ubuntu >= 18.04 (Not tested with earlier versions of Ubuntu or other OS)
* Python >= 3.6
* OpenCV >= 4.2
* Numpy >= 1.19


* YARP >= v3.3.2 
* ROS Noetic Ninjemys
* ROS 2 Humble Hawksbill **|** Galactic Geochelone **|** Foxy Fitzroy 
* PyZMQ 16.0, 17.1 and 19.0

To install Warpify:

```
python3 setup.py install
```

# Usage

Wrapyfi supports two patterns of communication: 
* **Publisher-Subscriber** (pub-sub): A publisher sends data to a subscriber accepting arguments and executing methods on the publisher's end.
e.g., with YARP


<table>
<tr>
<th> Without Wrapyfi </th>
<th> With Wrapyfi </th>
</tr>
<tr>
<td>
<sub>

```python
# Just your usual python class


class HelloWorld(object):
    
    
    
    
    def send_message(self):
        msg = input("Type your message: ")
        obj = {"message": msg}
        return obj,


hello_world = HelloWorld()



    

while True:
    my_message, = hello_world.send_message()
    print(my_message["message"])
```
    
</sub>
</td>
<td>
<sub>

```python
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class HelloWorld(MiddlewareCommunicator):
    @MiddlewareCommunicator.register("NativeObject", "yarp",
                                     "HelloWorld", 
                                     "/hello/my_message", 
                                     carrier="", should_wait=True)
    def send_message(self):
        msg = input("Type your message: ")
        obj = {"message": msg}
        return obj,


hello_world = HelloWorld()

LISTEN = True
mode = "listen" if LISTEN else "publish"
hello_world.activate_communication(hello_world.send_message, mode=mode)

while True:
    my_message, = hello_world.send_message()
    print(my_message["message"])
```
    
</sub>
</td>
</tr>
</table>

Run `yarpserver` from the command line. Now execute the python script above (with wrapyfi) twice setting `LISTEN = False` and `LISTEN = True`. You can now type with the publisher's command line and preview the message within the listener's


* **Request-Response** (req-res): A requester sends a request to a responder, which responds to the request in a synchronous manner.
e.g., with ROS

<table>
<tr>
<th> Without Wrapyfi </th>
<th> With Wrapyfi </th>
</tr>
<tr>
<td>
<sub>

```python
# Just your usual python class


class HelloWorld(object):
    
    
    
    
    def send_message(self, a=0, b=0):
        msg = input("Type your message: ")
        obj = {"message": msg, 
               "a": a, "b": b, "sum": a + b}
        return obj,


hello_world = HelloWorld()



    

while True:
    my_message, = hello_world.send_message(a=1, 
                                           b=2)
    print(my_message)
```
    
</sub>
</td>
<td>
<sub>

```python
from wrapyfi.connect.wrapper import MiddlewareCommunicator


class HelloWorld(MiddlewareCommunicator):
    @MiddlewareCommunicator.register("NativeObject", "ros",
                                     "HelloWorld", 
                                     "/hello/my_message", 
                                     carrier="", should_wait=True)
    def send_message(self, a=0, b=0):
        msg = input("Type your message: ")
        obj = {"message": msg, 
               "a": a, "b": b, "sum": a + b}
        return obj,


hello_world = HelloWorld()

LISTEN = True
mode = "request" if LISTEN else "reply"
hello_world.activate_communication(hello_world.send_message, mode=mode)

while True:
    my_message, = hello_world.send_message(a=1 if LISTEN else None, 
                                           b=2 if LISTEN else None)
    print(my_message["message"])
```
    
</sub>
</td>
</tr>
</table>




Run `roscore` from the command line. Now execute the python script above (with wrapyfi) twice setting `LISTEN = False` and `LISTEN = True`. You can now type within the server's command line and preview the message within the client's. 
Note that the server's command line will not show the message until the client's command line has been used to send a request. The arguments are passed from the client to the server and the server's response is passed back to the client.

<!--<img src="https://user-images.githubusercontent.com/4982924/144660266-42b00a00-72ee-4977-b5aa-29e3691321ef.gif" width="96%"/>-->

For more examples on usage, refer to the [usage documentation](docs/usage.md). Run scripts in the [examples directory](examples) for seeing Wrapyfi in action. 

# Supported Formats

## Middleware
* [x] **YARP**
* [x] **ROS**
* [x] **ROS 2**
* [x] **ZeroMQ** (TODO: proper should_wait trigger instead of dummy)

## Data Structures
* [x] **Numpy Array|Generic**
* [x] **Pytorch Tensor**
* [x] **Tensorflow 2 Tensor**
* [x] **JAX Tensor**
* [x] **MXNet Tensor**
* [x] **Paddlepaddle Tensor**
* [x] **Pandas Dataframe|Series**

## Serializers
* [x] **JSON**
* [ ] **msgpack**

## Image
* [x] **Numpy Array** (supports many libraries including [scikit-image](https://scikit-image.org/), [imageio](https://imageio.readthedocs.io/en/stable/), [Open CV](https://opencv.org/), [imutils](https://github.com/PyImageSearch/imutils), [matplotlib.image](https://matplotlib.org/stable/api/image_api.html), and [Mahotas](https://mahotas.readthedocs.io/en/latest/))
* [x] **Pillow Image**

# TODOS
* [x] **Support encapsulating wrapped calls to publishers and listeners**
* [ ] **Support wrapping for module functions**
* [x] **Support multiple class instances for functions set to publish [Experimental]**
