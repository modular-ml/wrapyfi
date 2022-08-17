# Wrapify

Wrapify is a middleware communication wrapper for transmitting data across nodes, without the need to
alter the operation pipeline of your python scripts. Wrapify introduces
a number of helper functions to make middleware integration possible without the need to learn an entire framework, just to parallelize your processes on 
multiple machines. 
Wrapify supports [Yarp](https://www.yarp.it/yarp_swig.html) and [ROS](http://wiki.ros.org/rospy).

To Wrapify a class, simply add the decorators describing the publisher and listener parameters. Wrapify imposes an object-oriented
requirement on your coding style: All wrapify compatible functions need to be defined within a class. 

## Installation

Before using Wrapify, Yarp or ROS must be installed. Follow the [Yarp installation guide](docs/yarp_install.md#installing-yarp).
Note that the iCub package is not needed for Wrapify to work and does not have to be installed if you do not intend on using the iCub robot.
For installation of ROS, follow the [ROS installation guide](docs/ros_install.md#installing-ros). 
We recommend installing ROS on conda using the [RoboStack](https://github.com/RoboStack/ros-noetic) environment.

#### compatibility
* Python >= 3.6
* OpenCV >= 4.2
* Yarp >= v3.3.2 
* ROS Noetic Ninjemys

To install Warpify:

```
python3 setup.py install
```

# Usage

<table>
<tr>
<th> Without Wrapify </th>
<th> With Wrapify </th>
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
from wrapify.connect.wrapper import MiddlewareCommunicator


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

Run `yarpserver` from the command line. Now execute the python script above (with wrapify) twice setting `LISTEN = False` and `LISTEN = True`. You can now type with the publisher's command line and preview the message within the listiner's

<img src="https://user-images.githubusercontent.com/4982924/144660266-42b00a00-72ee-4977-b5aa-29e3691321ef.gif" width="96%"/>

For more examples on usage, refer to the [usage documentation](docs/usage.md). Run scripts in the [examples directory](examples) for seeing Wrapify in action. 

# TODO
Visit the issues section for more details 
* [x] **Support ROS**
* [ ] **Support ROS 2**
* [ ] **Support ZeroMQ**
* [x] **Support Pytorch**
* [x] **Support Tensorflow 2**
* [ ] ~~Support Keras (TF 2)~~
* [ ] **Support Pandas dataframes**
* [ ] **Support Image formats: tensorflow, Pytorch, Scikit Image, ImageIO and Pillow**
* [ ] **Support msgpack as a serialization format**
* [x] **Support encapsulating wrapped calls to publishers and listeners**
* [ ] **Support wrapping for module functions**
* [ ] **Support multiple class instances for functions set to publish**
