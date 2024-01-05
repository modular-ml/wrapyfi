# Tutorial: Multiple Robot Control using the Mirroring and Forwarding Schemes

<p align="center">
    <video width="630" height="300" controls autoplay><source type="video/mp4" src="https://raw.githubusercontent.com/fabawi/wrapyfi/main/assets/tutorials/vid_demo_ex2-1.mp4"></video>
</p>

[Video: https://github.com/fabawi/wrapyfi/assets/4982924/a7ca712a-ffe8-40cb-9e78-b37d57dd27a4](https://github.com/fabawi/wrapyfi/assets/4982924/a7ca712a-ffe8-40cb-9e78-b37d57dd27a4)

This tutorial demonstrates how to use the Wrapyfi framework to run a facial expression recognition (FER) model on multiple robots. 
The model recognizes 8 facial expressions which are propagated to the Pepper and iCub robots. The expression categories are displayed by changing the Pepper robot's eye and shoulder LED colors---or 
*robotic facial expressions*---by changing the iCub robot's eyebrow and mouth LED patterns. The image input received by the model is acquired from the Pepper and iCub robots' cameras by simply 
**forwarding** the images to the facial expression recognition model (check out the [forwarding scheme](<../usage/User%20Guide/Communication%20Schemes.md#forwarding>) for more details on forwarding).
We also provide a simple application manager that handles the communication between the model and the robots. The application manager is responsible for forwarding images to the FER model, 
and transmitting recognized facial expressions to the robots. The application manager itself is composed of mirrored (check out the [mirroring scheme](<../usage/User%20Guide/Communication%20Schemes.md#mirroring>) 
instances running on one or several machines, depending on the configuration. 

## Methodology

<p align="center">
  <a id="figure-1"></a>
  <img width="460" src="https://raw.githubusercontent.com/fabawi/wrapyfi/main/assets/tutorials/wrapyfi_hri_ex2-1.png">
  <br>
  <em>Fig 1: Facial expression recognition for updating the affective cues on the Pepper and iCub robots.</em>
</p>

Siqueira et al. [(2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6037) presented a neural model for facial expression recognition, relying on an ensemble of convolutional branches with shared parameters. The model provides inference in real-time settings, owing to its relatively small number of parameters across ensembles, 
unimodal visual input, and non-sequential structure. For the last timestep $n$, a majority vote is cast on the output categories resulting from each ensemble branch $\text{e}_i$:

```math
\begin{align}
    \textbf{c}(f)_n &= \sum_{i=1}^{E} [\text{e}_i = f] \\
    \text{c}_n &= {arg\,max}_f \textbf{c}(f)_n
\end{align}
```

where $E=9$ is signifying the number of ensembles. The emotion category index is denoted by $f\in[1,8]$. The resulting $\textbf{c}(f)_n$ holds counts of the ensemble votes for each emotion category $f$ at $n$.

Given the model's sole reliance on static visual input, falsely recognized facial expressions lead to abrupt changes in the inferences. To mitigate sudden changes in facial expressions, we apply a mode smoothing filter to the last $N$ discrete predictions---eight emotion categories---where 
$N=6$ corresponding to the number of visual frames acquired by the model per second:

```math
\begin{align}
    \textbf{k}(f)_n &= \sum_{i=n-N+1}^{n} [\text{c}_i = f] \\
    \text{k}_n &= {arg\,max}_f \textbf{k}(f)_n
\end{align}
```

resulting in the emotion category $\text{k}_t$ being transmitted from the inference script running the facial expression recognition model to the application manager executed on **PC:A**. 
The application manager manages exchanges to and from the model and robot interfaces.

We execute the application on three to six machines, depending on the configuration:
* **PC:A** (*mware: YARP*): Running the application manager and forwarding messages to and from the FER model. 
* **S:1** (*mware: YARP*): Running the FER model and forwarding messages to and from the application manager.
* **PC:104** (*mware: YARP*): Running on the physical iCub robot (*only needed when running the physical robot*).
* **PC:ICUB** (*mware: YARP*): Running the iCub robot control workflow.
* **PC:PEPPER** (*mware: YARP, ROS*): Running the Pepper robot control workflow.
* **PC:WEBCAM** (*mware: YARP*): Running the webcam interface for acquiring images from the webcam (*only needed when running the simulated robot*).

**Note**: For this tutorial, **PC:ICUB**, **PC:WEBCAM**, and **PC:PEPPER** scripts are running on **PC:A** to simplify the process. However, they could also be executed on dedicated machines as long as the network configurations (`roscore` and `yarpserver` IP addresses) are set correctly. 

At least one of either two robot PCs (**PC:ICUB** and **PC:PEPPER**) must be running for the application to work. 
The webcam interface (**PC:WEBCAM**) is optional and is only needed if we want to acquire images from a webcam rather than a robots. 
We note that all machine scripts can be executed on a single machine, but we distribute them across multiple machines to demonstrate 
the flexibility of the Wrapyfi framework.

Images arrive directly from each robot's camera:
* The iCub robot image arrives from the left eye camera having a size of $320\times240$ pixels and is transmitted over YARP at 30 FPS. 
* The Pepper robot image arrives from the top camera having a size of $640\times480$ pixels and is transmitted over ROS at 24 FPS.
The image is directly forwarded to the facial expression model, resulting in a predicted expression returned to the corresponding robot's LED interface.

## Modifying the FER Model

To integrate Wrapyfi into the [facial expression recognition model](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks), we first need to modify the model to accept and return data from and to the robot interfaces.

This is achieved by using [Wrapyfi interfaces](https://github.com/modular-ml/wrapyfi-interfaces) which provide minimal examples of how to design the structure of templates and common interfaces, used for large-scale and complex applications. Templates and interfaces limit the types of data that can be transmitted. We can of course decide to transmit custom objects, something that Wrapyfi was designed to enable in the first place. However, in instances where we would like multiple applications to communicate and understand the information transmitted, a common structure *must* be introduced to avoid creating specific interfaces for each new application.

### Receiving Images from the Robot Interfaces

The model acquires images using OpenCV's VideoCapture class. We modify the model to receive images from the robot interfaces by replacing the VideoCapture class with a class that receives and returns images from any middleware supported by Wrapyfi. The modified class is defined in the
[Wrapyfi video interface](https://github.com/modular-ml/wrapyfi-interfaces/blob/main/wrapyfi_interfaces/io/video/interface.py). This interface is identical to the VideoCapture class, except that it receives and returns images from any middleware supported by Wrapyfi.

The interface is used to receive images from the robot interfaces by passing the middleware name and topic name to the interface constructor:

```python
from wrapyfi_interfaces.io.video.interface import VideoInterface

cap = VideoInterface("/icub/cam/left", mware="yarp")
```

In the example above, the interface receives images from the iCub robot's left eye camera over YARP. Note that here we replace the VideoCapture source with the topic name to which the robot's framework publishes the images. 
Similarly, we can receive images from the Pepper robot's top camera over ROS by passing the topic name to the interface constructor:

```python
cap = VideoInterface("/pepper/camera/front/camera/image_raw", mware="ros")
```

Getting the return value from the interface is identical to the VideoCapture class:

```python
ret, frame = cap.read()
```

with every call to `cap.read()` returning a boolean value `ret` indicating whether the frame was successfully read, and the image `frame` itself. 


### Sending the Recognized Expression to the Robot Interfaces
The [Facial Expression Message Template](https://github.com/modular-ml/wrapyfi-interfaces/blob/main/wrapyfi_interfaces/templates/facial_expressions.py) provided as part of the [Wrapyfi interfaces](https://github.com/modular-ml/wrapyfi-interfaces) collection, allows for standardized transmission of information relating to affect. This template is similar in operation to other interfaces, where instead of wrapping methods with the Wrapyfi registry decorator, we simply call the method with arguments specifying *"what"* should be transmitted and *"where/how"* (as in the port/topic address, communication pattern, middleware).

We first import the template and instantiate it:

```python
from wrapyfi_interfaces.templates.facial_expressions import FacialExpressionsInterface

_FACIAL_EXPRESSION_BROADCASTER = FacialExpressionsInterface(facial_expressions_port_out=facial_expressions_port,
    mware_out=facial_expressions_mware, facial_expressions_port_in="")
```

Setting the `facial_expressions_port_out` and `mware_out` arguments tells the template that it should activate its communication in `publish` mode, meaning that it would be transmitting emotion categories rather than receiving them. In this case, we specify the receiving port as empty, since receiving affective signals is not needed.

Next, we must send the prediction signal (the emotion category, scores, emotion continuous---arousal and valence, emotion index, etc.). This is done by calling `transmit_emotion()` everytime a prediction is made:

```python
prediction, = _FACIAL_EXPRESSION_BROADCASTER.transmit_emotion(*(_predict(input_face, device)),
    facial_expressions_port=facial_expressions_port, _mware=facial_expressions_mware)
```

Where the prediction dictionary is transmitted over the middleware and returned as `prediction` from the method call. Now, any template called from another instance of the same application or any other application subscribed to the specified port/topic on the same middleware within the network should be able to receive the prediction dictionary. This allows the robot or any controller to receive the values predicted by the model at any step in time, as long as the model ESR9 is running. 

## Pre-requisites:

**Note**: The following installation instructions are compatible with **Ubuntu 18-22** and are not guaranteed to work on other distributions or operating systems.

* Install [Wrapyfi](https://wrapyfi.readthedocs.io/en/latest/readme_lnk.html#installation) on all machines (excluding **PC:104**)
* Install [PyTorch](https://pytorch.org/get-started/locally/) for running the facial expression recognition model on **S:1**
* Install the [ESR9 FER model with Wrapyfi](https://github.com/modular-ml/wrapyfi-examples_ESR9) requirements on **S:1**

Throughout this tutorial, we assume that all repositories are cloned into the `$HOME\Code` directory.
**Wrapyfi should also be cloned into the `$HOME\Code` directory in order to access the examples.**

Additionally, cloning the [Wrapyfi interfaces](https://github.com/modular-ml/wrapyfi-interfaces) repository on all machines (excluding **PC:104**) is needed 
since it provides dedicated interfaces for communicating with the robots, acquiring and publishing webcam images, 
and providing message structures for standardizing exchanges between applications: 

```bash
cd $HOME/Code
git clone https://github.com/modular-ml/wrapyfi-interfaces.git
```

and add it to the `PYTHONPATH` environment variable:

```bash
export PYTHONPATH=$PYTHONPATH:$HOME/Code/wrapyfi-interfaces
```

### When Using the Pepper Robot with NAOqi 2.5:

**Note**: Installation instructions apply to **PC:PEPPER**

* [ROS](https://wiki.ros.org/noetic) and Interfaces:
  * Install [ROS Noetic](http://wiki.ros.org/ROS/Installation) 
  **or** 
  [Robostack bundling of ROS Noetic in a mamba or micromamba environment](https://robostack.github.io/GettingStarted.html)
  * Install the camera info manager for the Pepper camera on local system: `sudo apt install ros-noetic-camera-info-manager` 
  **or** 
  within a Robostack env: `micromamba install -c robostack ros-noetic-camera-info-manager`
  * Activate and source ROS on local system: `source /opt/ros/noetic/setup.bash`
  **or**
  activate the Robostack env: `micromamba activate ros_env`
  * Clone the [Pepper Camera](https://github.com/modular-ml/pepper_camera) package:
  ```bash
  cd $HOME/Code
  git clone https://github.com/modular-ml/pepper_camera.git
  ```
  * Install the Pepper Camera dependencies on local system: `sudo apt install libgstreamer1.0-dev gstreamer1.0-tools`
  **or**
  within a Robostack env: `micromamba install gst-plugins-base gst-plugins-good gstreamer -c conda-forge`
  * Create a ROS workspace and link the Pepper Camera resources into it:
  ```bash
  mkdir -p $HOME/pepper_ros_ws/src
  cd $HOME/pepper_ros_ws
  ln -s $HOME/Code/pepper_camera src/pepper_camera
  ```
  * Compile the ROS node using catkin:
  ```bash
  catkin_make
  ```
  
* [Docker with NAOqi & ROS Kinetic - Python 2.7](https://github.com/modular-ml/pepper-ros-docker):
  * Install [Docker](https://docs.docker.com/engine/install/ubuntu/)
  * Clone the [Pepper ROS Docker](https://github.com/modular-ml/pepper-ros-docker) repository:
  ```bash
  cd $HOME/Code
  git clone https://github.com/modular-ml/pepper-ros-docker.git
  ```
  * Build the Pepper ROS Docker image:
  ```bash
  cd pepper-ros-docker
  docker build . -t minimal-pepper-ros-driver
  ```
  
### When using the iCub Robot:

**Note**: Installation instructions apply to **PC:ICUB**. They can also be followed for **PC:A**, **S:1**, **PC:WEBCAM**, and **PC:PEPPER**, however, only YARP with Python bindings is needed for these machines. If these machines have their required packages and Wrapyfi installed inside a mamba or micromamba environment, then installing the following within the environment should suffice: `micromamba install -c robotology yarp`

* Install [YARP](https://yarp.it/latest//install_yarp_linux.html) and [iCub Software](https://icub-tech-iit.github.io/documentation/sw_installation/) on local system following our [recommended instructions](https://wrapyfi.readthedocs.io/en/latest/yarp_install_lnk.html)
**or**
within a mamba or micromamba environment using the [robotology-superbuild](https://github.com/robotology/robotology-superbuild/blob/master/doc/conda-forge.md): 
* Activate and source YARP ([step 5 in installing YARP](https://wrapyfi.readthedocs.io/en/latest/yarp_install_lnk.html#installing-yarp)) on local system
**or**
activate the robotology-superbuild env: `micromamba activate robotologyenv`
* Install the [Pexpect](https://pexpect.readthedocs.io/en/stable/) Python package: `pip install pexpect`

## Running the Application

<details>

  <summary><b><font color="green">Easy</font>: iCub simulation only; running all scripts on a single machine</b></summary>

  Here we mirror the facial expressions of an actor facing a webcam on a simulated iCub robot. The images from the webcam are streamed to the ESR9 [(Siqueira et al., 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6037) FER model, which then classifies their facial expressions and returns the predicted class to the application controller (robot workflow controller). The controller transmits the readings to the iCub interface and displays an approximated facial expression on the robot's face.
  
  ### Preparing the iCub robot (in simulation)
  
  Start the yapserver to enable communication with the iCub robot (on any machine):
  
  ```bash
  yarpserver
  ```

  Start the iCub simulator (on **PC:ICUB**):
    
  ```bash
  iCub_SIM
  ```

  The facial expressions shown on the iCub's face are not enabled by default when running the iCub simulator, so we need to start the `iCubFaceExpressions` module to enable them (on **PC:ICUB**):
  
  ```bash 
  simFaceExpressions
  ```

  Start the iCub emotion interface to receive the facial expressions on a specific port/topic (on **PC:ICUB**):
  
  ```bash
  emotionInterface --name /icubSim/face/emotions --context faceExpressions --from emotions.ini
  ```

  Connect the iCub simulator ports to the iCub emotion interface (on **PC:ICUB**):
  
  ```bash
  yarp connect /face/eyelids /icubSim/face/eyelids
  yarp connect /face/image/out /icubSim/texture/face
  yarp connect /icubSim/face/emotions/out /icubSim/face/raw/in
  ```
  ### Running the iCub interface
  
  Start the iCub interface to receive the facial expressions from the application controller and activate the facial expressions on the iCub robot (on **PC:ICUB**):
  
  ```bash 
  cd $HOME/Code/wrapyfi-interfaces
  python wrapyfi_interfaces/robots/icub_head/interface.py \
  --simulation --get_cam_feed \
  --control_expressions \
  --facial_expressions_port "/control_interface/facial_expressions_icub"
  ```
  
  Start the camera interface to receive images from the webcam and forward them to the application controller (on **PC:WEBCAM**):

  ```bash
  cd $HOME/Code/wrapyfi/examples/applications
  python wrapyfi_interfaces/io/video/interface.py --mware yarp --cap_source "0" --fps 30 --cap_feed_port "/control_interface/image_webcam" --img_width 320 --img_height 240 --jpg
  ```
  
  Start two mirrored instances of the application controller (on **PC:A** and **PC:ICUB**, respectively):

  1) The first instance is responsible for running the application workflow (on **PC:A**):
  
  ```bash
  cd $HOME/Code/wrapyfi/examples/applications
  WRAPYFI_DEFAULT_COMMUNICATOR="yarp" python affective_signaling_multirobot.py --wrapyfi_cfg wrapyfi_configs/affective_signaling_multirobot/COMP_mainpc.yml --cam_source webcam
  ```

  2) The second instance is responsible for running the robot (iCub) control workflow (on **PC:ICUB**):
  
  ```bash
  cd $HOME/Code/wrapyfi/examples/applications
  WRAPYFI_DEFAULT_COMMUNICATOR="yarp" python affective_signaling_multirobot.py --wrapyfi_cfg wrapyfi_configs/affective_signaling_multirobot/OPT_icubpc.yml --cam_source webcam
  ```

  **Note**: running two instances is not necessary if we configure a single script to handle all exchanges; however, 
  we do so to separate the application workflow from the robot control workflows. In this example, where we run a single 
  robot, the utility of such separation is not apparent. If we were to merge the workflows in the main configuration 
  `COMP_mainpc.yml` file, then we must also run the workflow for **robot A** when wanting to run **robot B** only.
  
  Run the ESR9 FER model, acquiring images from the webcam and forwarding the recognized expression to the application controller (on **S:1**):
  
  ```bash
  cd $HOME/Code/wrapyfi-examples_esr9/
  export PYTHONPATH=$HOME/Code/wrapyfi-interfaces:$PYTHONPATH
  python main_esr9.py webcam -w "/control_interface/image_esr9" -d -s 2 -b --frames 10 --max_frames 10 --video_mware yarp --facial_expressions_mware yarp --facial_expressions_port "/control_interface/facial_expressions_esr9" --face_detection 3 --img_width 320 --img_height 240 --jpg
  ```

  **Outcome**: Make sure you are facing the webcam and you should now be able to see the simulated iCub robot changing his facial expressions, corresponding to your own.
</details>

<details>

  <summary><b><font color="orange">Intermediate</font>: iCub & Pepper; running scripts on multiple machine</b></summary>
  
  Here we mirror the facial expressions of an actor facing the Pepper or iCub robot camera on both (physical) robots. The images from the chosen camera are streamed to the ESR9 [(Siqueira et al., 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6037) FER model, which then classifies their facial expressions and returns the predicted class to the application controller (robot workflow controller). The controller transmits the readings to the iCub and Pepper robot interfaces, displays an approximated facial expression on the iCub robot's face, and triggers a color change on the Pepper robot's eye and shoulder LEDs.
  
  ### Preparing the iCub robot

  * Connect the iCub robot to the power supply and switch it on (please follow the instructions specific to your iCub robot)
  * Connect your iCub robot's (**PC:104**) ethernet cable to a network switch attached to all other machines (excluding **PC:WEBCAM** which is not needed in this setup)
  
  Start the `yapserver` to enable communication with the iCub robot (on any machine):
  
  ```bash
  yarpserver
  ```

  **Note**: Ensure every PC is configured to detect `yarpserver`. Assuming the `yarpserver` is running on a machine with an IP `<IP yarpserver>`:
  
  ```bash
  yarp detect <IP yarpserver> 10000
  ```

  Initialize and configure the iCub camera device on a specific port/topic (on **PC:104**):

  ```bash
  yarpdev --from camera/ServerGrabberDualDragon.ini --split true --framerate 30
  ```

  Initialize and configure the iCub emotion device on a specific port/topic (on **PC:104**):

  ```bash
  yarpdev --name /icub/face/raw --device serial --subdevice serialport --context faceExpressions --from serialport.ini
  ```

  Start the iCub emotion interface to receive the facial expressions on a specific port/topic (on **PC:104**):

  ```bash
  emotionInterface --name /icub/face/emotions --context faceExpressions --from emotions.ini
  ```
  Connect the input/output ports for expression reading and writing (on **PC:104**):
  
  ```bash
  yarp connect /icub/face/emotions/out /icub/face/raw/in
  ```

  ### Preparing the Pepper robot

  * Connect an ethernet cable to the back of the Pepper robot's head
  * Connect the other end of the ethernet cable to a network switch attached to all other machines (excluding **PC:WEBCAM** which is not needed in this setup)
  * Switch on the Pepper Robot
  * On initialization completion, press the chest button on the Pepper robot for him to speak out its current IP. This IP will be referred to as `<IP Pepper>` 

  Build the Pepper ROS workspace and start the `roscore` to enable communication with the Pepper robot (on **PC:PEPPER**):

  ```bash
  cd $HOME/pepper_ros_ws
  catkin build
  roscore
  ```

  **Note**: Ensure the Pepper ROS Docker container (and any other machine using ROS if manual changes to the configuration files are made) is configured to detect the `roscore` URI. Assuming the `roscore` is running on **PC:PEPPER** with an IP `<IP roscore>`:
  
  ```bash
  export ROS_MASTER_URI=<IP roscore>
  ```

  If the Pepper ROS Docker image was built under the name `minimal-pepper-ros-driver:latest`, start the container (on **PC:PEPPER**):

  ```bash
  docker ps -a
			
  # If no container exists:
  docker run -it --network host --name pepperdock minimal-pepper-ros-driver:latest
		
  # If a container exists but is 'exited':
  docker start pepperdock
  ```

  Launch the Pepper robot's interfaces within the container (on **PC:PEPPER**):
  
  ```bash
  docker exec -it pepperdock bash -i
		export ROS_MASTER_URI=http://<IP roscore>:11311
		roslaunch pepper_extra pepper_wrapyfi.launch ip:=<IP Pepper> 
   ```

  Call the ROS services on the Pepper robot to start them within the docker container. The robot should transition to an idle mode without movement and speak out (on **PC:PEPPER**):

  ```bash
  docker exec -it pepperdock bash -i
		export ROS_MASTER_URI=http://<IP roscore>:11311
		rosservice call /pepper/pose/idle_mode "{idle_enabled: true, breath_enabled: false}"
		rosservice call /pepper/pose/home
		rosservice call /pepper/speech/say "{text: 'hello and welcome, my name is pepper', wait: false}"
  ```

  ### Running the robot interfaces
  
  Start the iCub interface to receive the facial expressions from the application controller and activate the facial expressions on the iCub robot (on **PC:ICUB**):
  
  ```bash 
  cd $HOME/Code/wrapyfi-interfaces
  python wrapyfi_interfaces/robots/icub_head/interface.py \
  --get_cam_feed \
  --control_expressions \
  --facial_expressions_port "/control_interface/facial_expressions_icub"
  ```
  
  Start the Pepper interface to receive the facial expressions from the application controller and enable the LED color changes on the Pepper robot (on **PC:PEPPER**):
  
  ```bash
  source $HOME/pepper_ros_ws/devel/setup.bash
  cd $HOME/Code/wrapyfi-interfaces
  python wrapyfi_interfaces/robots/pepper/interface.py \
  --get_cam_feed \
  --control_expressions \
  --facial_expressions_port "/control_interface/facial_expressions_pepper"
  ```

  Start three mirrored instances of the application controller (on **PC:A**, **PC:ICUB**, and **PC:PEPPER**, respectively):

  1) The first instance is responsible for running the application workflow (on **PC:A**):
  
  ```bash
  cd $HOME/Code/wrapyfi/examples/applications
  WRAPYFI_DEFAULT_COMMUNICATOR="yarp" python affective_signaling_multirobot.py --wrapyfi_cfg wrapyfi_configs/affective_signaling_multirobot/COMP_mainpc.yml --cam_source pepper
  ```

  2) The second instance is responsible for running the robot (iCub) control workflow (on **PC:ICUB**):
  
  ```bash
  cd $HOME/Code/wrapyfi/examples/applications
  WRAPYFI_DEFAULT_COMMUNICATOR="yarp" python affective_signaling_multirobot.py --wrapyfi_cfg wrapyfi_configs/affective_signaling_multirobot/OPT_icubpc.yml --cam_source pepper
  ```

  3) The third instance is responsible for running the robot (Pepper) control workflow (on **PC:PEPPER**):
  
  ```bash
  cd $HOME/Code/wrapyfi/examples/applications
  WRAPYFI_DEFAULT_COMMUNICATOR="yarp" python affective_signaling_multirobot.py --wrapyfi_cfg wrapyfi_configs/affective_signaling_multirobot/OPT_pepperpc.yml --cam_source pepper
  ```

  **Note**: The `--cam_source` argument can be set to either `icub` or `pepper`, defining where from the image arrives. Switching the camera source requires minimal changes to the control workflow instances and does not affect the FER model since the camera image is forwarded from the source to a dedicated topic/port to which the FER subscribes.
  
  Run the ESR9 FER model, acquiring images from the webcam and forwarding the recognized expression to the application controller (on **S:1**):
  
  ```bash
  cd $HOME/Code/wrapyfi-examples_esr9/
  export PYTHONPATH=$HOME/Code/wrapyfi-interfaces:$PYTHONPATH
  python main_esr9.py webcam -w "/control_interface/image_esr9" -d -s 2 -b --frames 10 --max_frames 10 --video_mware yarp --facial_expressions_mware yarp --facial_expressions_port "/control_interface/facial_expressions_esr9" --face_detection 3 --img_width 320 --img_height 240 --jpg
  ```

  **Outcome**: Make sure you are facing the right camera (Pepper or iCub) and you should now be able to see the robots changing their facial expressions (iCub) or LED colors (Pepper) corresponding to your facial expressions.
</details>
