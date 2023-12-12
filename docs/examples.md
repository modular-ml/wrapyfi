# Basic Examples

## [Hello World](https://wrapyfi.readthedocs.io/en/latest/examples/examples.html#module-examples.hello_world)
This example shows how to use the MiddlewareCommunicator to send and receive messages. It can be used to test the functionality of the middleware using the PUB/SUB pattern and the REQ/REP pattern. The example can be run on a single machine or on multiple machines. In this example (as with all other examples), the communication middleware is selected using the `--mware` argument. The default is ZeroMQ, but YARP, ROS, and ROS 2 are also supported.


# Communication Schemes

## [Mirroring](https://wrapyfi.readthedocs.io/en/latest/examples/examples.communication_schemes.html#module-examples.communication_schemes.mirroring_example)
This script demonstrates the capability to mirror messages using the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB and REQ/REP patterns, allowing message publishing, listening, requesting, and replying functionalities between processes or machines.

## [Forwarding](https://wrapyfi.readthedocs.io/en/latest/examples/examples.communication_schemes.html#module-examples.communication_schemes.forwarding_example)
This script demonstrates message forwarding using the MiddlewareCommunicator within the Wrapyfi library. The communication follows chained forwarding through two methods, enabling PUB/SUB pattern that allows message publishing and listening functionalities between processes or machines.

## [Channeling](https://wrapyfi.readthedocs.io/en/latest/examples/examples.communication_schemes.html#module-examples.communication_schemes.channeling_example)
This script demonstrates message channeling through three different middleware (A, B, and C) using the MiddlewareCommunicator within the Wrapyfi library. It allows message publishing and listening functionalities between processes or machines.

# Communication Patterns

[//]: # (## [PUB/SUB]&#40;https://wrapyfi.readthedocs.io/en/latest/examples/examples.communication_patterns.html#module-examples.communication_patterns.pub_sub_example&#41;)

[//]: # (This script demonstrates the capability to publish and subscribe to messages using the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern, allowing message publishing and listening functionalities between processes or machines.)

## [REQ/REP](https://wrapyfi.readthedocs.io/en/latest/examples/examples.communication_patterns.html#module-examples.communication_patterns.request_reply_example)
This script demonstrates the capability to request and reply to messages using the MiddlewareCommunicator within the Wrapyfi library. The communication follows the REQ/REP pattern, allowing message requesting and replying functionalities between processes or machines.

# Custom Messages

## [ROS Message](https://wrapyfi.readthedocs.io/en/latest/examples/examples.custom_msgs.html#module-examples.custom_msgs.ros_message_example)
This script demonstrates the capability to transmit ROS messages, specifically `geometry_msgs/Pose` and `std_msgs/String`, using the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern allowing message publishing and listening functionalities between processes or machines.

## [ROS Parameter](https://wrapyfi.readthedocs.io/en/latest/examples/examples.custom_msgs.html#module-examples.custom_msgs.ros_parameter_example)
This script demonstrates the capability to transmit ROS properties, specifically using the Properties message, using the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern allowing property publishing and listening functionalities between processes or machines.

## [ROS2 Message](https://wrapyfi.readthedocs.io/en/latest/examples/examples.custom_msgs.html#module-examples.custom_msgs.ros2_message_example)
This script demonstrates the capability to transmit ROS 2 messages, specifically `geometry_msgs/Pose` and `std_msgs/String`, using the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern allowing message publishing and listening functionalities between processes or machines.

# Plugins (Encoders)

## [Astropy](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.astropy_example)
A message publisher and listener for native Python objects and Astropy Tables (external plugin).

## [Pint](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.pint_example)
A message publisher and listener for native Python objects and Pint Quantities.

## [Pillow](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.pillow_example)
A message publisher and listener for PIL (Pillow) images.

## [DASK](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.dask_example)
A message publisher and listener for native Python objects and Dask Arrays/Dataframes.

## [Numpy and pandas](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.numpy_pandas_example)
A message publisher and listener for native Python objects, NumPy Arrays, and pandas Series/Dataframes.

## [PyArrow](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.pyarrow_example)
A message publisher and listener for native Python objects and PyArrow arrays.

## [xarray](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.xarray_example)
A message publisher and listener for native Python objects and xarray DataArrays.

## [Zarr](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.zarr_example)
A message publisher and listener for native Python objects and Zarr arrays/groups.

## [JAX](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.jax_example)
A message publisher and listener for native Python objects and JAX arrays.

## [MXNet](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.mxnet_example)
A message publisher and listener for native Python objects and MXNet tensors.

## [PaddlePaddle](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.paddlepaddle_example)
A message publisher and listener for native Python objects and PaddlePaddle tensors.

## [PyTorch](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.pytorch_example)
A message publisher and listener for native Python objects and PyTorch tensors.

## [TensorFlow](https://wrapyfi.readthedocs.io/en/latest/examples/examples.encoders.html#module-examples.encoders.tensorflow_example)
A message publisher and listener for native Python objects and TensorFlow tensors.

# Robots

## [iCub Head](https://wrapyfi.readthedocs.io/en/latest/examples/examples.robots.html#module-examples.robots.icub_head)
This script demonstrates the capability to control the iCub robot’s head and view its camera feed using the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern, allowing for message publishing and listening functionalities between processes or machines.

# Sensors

## [Camera and Microphone](https://wrapyfi.readthedocs.io/en/latest/examples/examples.sensors.html#module-examples.sensors.cam_mic)
This script demonstrates the capability to transmit audio and video streams using the MiddlewareCommunicator within the Wrapyfi library. The communication follows the PUB/SUB pattern allowing message publishing and listening functionalities between processes or machines.
