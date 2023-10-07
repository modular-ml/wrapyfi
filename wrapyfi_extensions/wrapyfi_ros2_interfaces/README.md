# Compiling ROS2 interfaces

**WARNING**: These instructions are located in 
https://github.com/fabawi/wrapyfi/blob/master/wrapyfi_extensions/wrapyfi_ros2_interfaces

To run the Wrapyfi ROS2 interfaces, you need to compile the ROS2 interfaces. To do so, you need to have ROS2 installed on your system. You can find the installation instructions [here](https://docs.ros.org/en/humble/Installation.html).

## Prerequisites

- ROS2 Galactic/Humble
- Python 3.6

## Compiling

1. Copy the `wrapyfi_ros2_interfaces` folder to your ROS2 workspace (assumed to be `~/ros2_ws`).

    ```bash
    # from the current directory 
    cd ../
    cp -r wrapyfi_ros2_interfaces ~/ros2_ws/src
    
    ```

2. Compile the ROS2 interfaces:
    
    ```bash
    cd ~/ros2_ws
    colcon build --packages-select wrapyfi_ros2_interfaces
    
    ```
    
    **Note**: If the wrong version of Python is used, the compilation will fail. Make sure that the correct version of cmake 
    is used by modifying the `cmake_minimum_required` version in the `~/ros2_ws/src/wrapyfi_ros2_interfaces/CMakeLists.txt` file:
    
    ```cmake
    # CMakeLists.txt
    cmake_minimum_required(VERSION 3.5)
    # ...
    ```
    
    Replacing VERSION 3.5 with the correct version of cmake.

3. Source the ROS2 workspace:

    ```bash
    source ~/ros2_ws/install/setup.bash
    ```

4. Verify that the ROS2 Native object service interface is compiled:
    
    ```bash
    ros2 interface show wrapyfi_ros2_interfaces/srv/ROS2NativeObjectService
    ```
    
    Which should output:
    
    ```bash
    string request
    ---
    string response
    ```

5. Verify that the ROS2 Image service interface is compiled:
        
    ```bash
    ros2 interface show wrapyfi_ros2_interfaces/srv/ROS2ImageService
    ```
    
    Which should output:
    
    ```bash
    string request
    ---
    sensor_msgs/Image response
        std_msgs/Header header
            builtin_interfaces/Time stamp
                int32 sec
                uint32 nanosec
            string frame_id
                                     # Header frame_id should be optical frame of camera
                                     # origin of frame should be optical center of cameara
                                     # +x should point to the right in the image
                                     # +y should point down in the image
                                     # +z should point into to plane of the image
                                     # If the frame_id here and the frame_id of the CameraInfo
                                     # message associated with the image conflict
                                     # the behavior is undefined
        uint32 height
        uint32 width
        string encoding
                              # taken from the list of strings in include/sensor_msgs/image_encodings.hpp
        uint8 is_bigendian
        uint32 step
        uint8[] data
    
    ```
   
     Run your Wrapyfi enabled script from the same terminal. Now you can use the REQ/REP pattern (server/client) in Wrapyfi
     