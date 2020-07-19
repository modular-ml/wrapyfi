# Wrapify

Wrapify is a middleware communication wrapper for effortlessly transmitting data across nodes, without the need to
alter the operation pipeline of your python scripts. Wrapify introduces
a number of helper functions to make middleware integration possible without the need to learn an entire framework, just to parallelize your processes on 
multiple machines. 
Wrapify is currently compatible with the[Yarp python bindings](https://www.yarp.it/yarp_swig.html) package only, but can be extended to support other middleware
platforms.

To Wrapify a class, simply add the decorators describing the publisher and listener parameters. Wrapify imposes an object-oriented
requirement on your coding style: All wrapify compatible functions need to be defined within a class. 

## Installation

Before using Wrapify, Yarp must be installed. Follow the [Yarp installation guide](docs/yarp_install.md#installing-yarp).
Note that the iCub package is not needed for Wrapify to work and does not have to be installed if you do not intend on using the iCub robot.

#### compatibility
* Python >= 3.6
* Yarp >= v3.3.2 
* OpenCV >= 4.2

To install Warpify:

```
python3 setup.py install
```

# Usage

For examples on usage, refer to the [usage documentation](docs/usage.md). Run scripts in the [examples directory](examples) for seeing Wrapify in action. 

# TODO
Visit the issues section for more details 
* **Support ROS**
* **Support encapsulating wrapped calls to publishers and listeners**
* **Support wrapping for module functions**
* **Support multiple class instances for functions set to publish**