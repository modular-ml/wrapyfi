# YARP + iCub Setup and Instructions
To run the iCub simulator, the YARP framework as well as the iCub simulator need to be installed. Before cloning YARP and iCub, checkout the [compatible versions](http://wiki.icub.org/wiki/Software_Versioning_Table).
For this guide, we install YARP v3.3.2 and iCub (icub-main) v1.16.0. 

## Installing YARP
Several instructional tutorials are available for installing YARP and its python bindings.
The installation process described here works on Ubuntu 20.04 and Python 3.8.
Download the YARP source and install following the [official installation documentation](https://www.yarp.it/install_yarp_linux.html) (do not use the precompiled binaries).
Set the environment variable ```YARP_ROOT``` to the YARP source location:

```export YARP_ROOT=<YOUR YARP LOCATION>```

Once YARP has been successfully installed, we proceed by installing the python bindings
1. Go to the YARP directory ```cd $YARP_ROOT/bindings```
2. Create a build directory ```mkdir build```
3. Compile the Python bindings (Assuming Python3.8 is installed):
    ```
   cmake -DYARP_COMPILE_BINDINGS:BOOL=ON -DCREATE_PYTHON:BOOL=ON -DYARP_USE_PYTHON_VERSION=3.8 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so -DCMAKE_INSTALL_PYTHONDIR=lib/python3.8/dist-packages -DPYTHON_EXECUTABLE=/usr/bin/python3.8 ..
   ``` 
   **NOTE**: Specify the following env. vars if Python3 libraries cannot be found (make sure that the locations exist):
   ```
   export PYTHON_INCLUDE_DIR=/usr/include/python3.8
   export PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so
   ```
4. Install the Python bindings
    ```
   make
   sudo make install
    ```
   
5. Per [official instructions](https://www.yarp.it/yarp_swig.html), the ```PYTHONPATH``` and ```LD_LIBRARY_PATH``` 
should be set to the following (assuming you are in the build directory):
    ```
   export PYTHONPATH=$PWD:$PYTHONPATH
   setenv LD_LIBRARY_PATH $PWD:$LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
    ```
   however, setting ```PYTHONPATH``` to the directory where the ```.so``` file exists only seemed to work:
    ```
   export PYTHONPATH=$YARP_ROOT/bindings/build/lib/python:$PYTHONPATH
    ```
   You can add the ```export PYTHON```... command to your ```~/.bashrc``` to avoid setting it again for a new session. 
   
## Installing iCub

Install from the source by following the [official installation documentation](http://wiki.icub.org/wiki/Linux:Installation_from_sources).

## Running the Interface (in Simulation)

1. Start the YARP server from the console 
    
    ```yarpserver```
    
2. Start the iCub simulator in another console 
    
    ```iCub_SIM```
    
3. To issue instructions for controlling the iCub's head from this Python-based interface, simply start the following script from this repository's root directory: 

    ```
   python3 examples/robots/icub_head.py \
    --simulation --get_cam_feed --control_head \
   --set_head_eye_coordinates \
   --head_eye_coordinates_port "/control_interface/head_eye_coordinates" \
   --control_expressions --set_facial_expressions \
   --facial_expressions_port "/emotion_interface/facial_expression"
   ```
    
    You can also manually control the joints by using the YARP's built-in motor controller GUI 

    ```yarpmotorgui```
    
    This interface uses YARP for controlling the iCub, however, communicating control from other devices is 
    limited to YARP. The default communcation (set to yarp) middleware can be overridden through environment variables set 
    before running the `interface.py` file e.g.:
    
   ```
   export ICUB_DEFAULT_MWARE=ros
   ```
   
   or passing `--mware` argument to the `interface.py` file e.g.:
   
   ```
    python3 yaw/robots/icub_head/interface.py --mware ros ...
   ```
   
## Facial Expressions in Simulation

The facial expressions do not run out-of-the-box. For compatibility with the `icub_head.py`, the following scripts need to 
be executed after starting the `iCub_SIM`:

1. Start the simulator face expression interface:

   `simFaceExpressions`

2. Start the emotion interface:

   `emotionInterface --name /icubSim/face/emotions --context faceExpressions --from emotions.ini`

3. Connect the emotion interface ports:

    ```
   yarp connect /face/eyelids /icubSim/face/eyelids
   yarp connect /face/image/out /icubSim/texture/face
   yarp connect /icubSim/face/emotions/out /icubSim/face/raw/in
    ```
   
## Running YARP on multiple machines
1. Run the YARP server on the host machine:
    
    ```yarpserver --ip <YOUR NETWORK IP> --socket 10000```
    
2. Detect the YARP server on the client machine(s):
    
    ```
    yarp conf <YOUR NETWORK IP> 10000
    # alternatively
    # yarp detect --write
    ```
