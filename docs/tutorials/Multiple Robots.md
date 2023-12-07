# Tutorial: Multiple Robot Control using the Forwarding Scheme

<p align="center">
    <video width="630" height="300" controls autoplay><source type="video/mp4" src="https://raw.githubusercontent.com/fabawi/wrapyfi/main/assets/tutorials/vid_demo_ex2-1.mp4"></video>
</p>

[Video: https://github.com/fabawi/wrapyfi/assets/4982924/a7ca712a-ffe8-40cb-9e78-b37d57dd27a4](https://github.com/fabawi/wrapyfi/assets/4982924/a7ca712a-ffe8-40cb-9e78-b37d57dd27a4)

This tutorial demonstrates how to use the Wrapyfi framework to run a facial expression recognition model on multiple robots. The facial expression recognition model is executed on four machines, each having a GPU. 
The model recognizes 8 facial expressions which are propagated to the Pepper and iCub robots. The expression categories are displayed by changing the Pepper robot's eye and shoulder LED colors---or 
\textit{robotic facial expressions}---by changing the iCub robot's eyebrow and mouth LED patterns. The image input received by the model is acquired from the Pepper and iCub robots' cameras by simply 
**forwarding** the images to the facial expression recognition model (check out the [forwarding scheme](<../usage/User%20Guide/Communication%20Schemes.md#forwarding>) for more details on forwarding).

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

resulting in the emotion category $\text{k}_t$ being transmitted from the inference script running the facial expression recognition model to the managing script executed on **PC:A**. The managing script is responsible for forwarding data to and from the model and robot interfaces.
We execute the inference script on four machines. The shared layer weights are loaded on an NVIDIA GeForce GTX 970 (denoted by **PC:A** in [**Figure 1**](#figure-1)) with 4 GB VRAM. Machines **S:1**, **S:2**, and **S:3** share similar specifications, each with an NVIDIA GeForce GTX 1050 Ti having 4GB VRAM. 
We distribute nine ensembles among the three machines in equal proportions and broadcast their latent representation tensors using ZeroMQ. The PyTorch-based inference script is executed on **PC:A**, **S:1**, **S:2**, and **S:3**, all having their tensors mapped to a GPU. 

Depending on the experimental condition, images arrive directly from each robot's camera:
* The iCub robot image arrives from the left eye camera having a size of $320\times240$ pixels and is transmitted over YARP at 30 FPS. 
* The Pepper robot image arrives from the top camera having a size of $640\times480$ pixels and is transmitted over ROS at 24 FPS.
The image is directly forwarded to the facial expression model, resulting in a predicted emotion returned to the corresponding robot's LED interface.

## Pre-requisites:
* Installing [Wrapyfi](<../usage/Installation.md>)
* Installing [PyTorch](https://pytorch.org/get-started/locally/) for running the facial expression recognition model
* **when using the Pepper robot**:
  * [ROS](http://wiki.ros.org/ROS/Installation)
  * [DOCKER with Naoqi]
* **when using the iCub robot**:
  * [YARP](https://www.yarp.it/install.html)
  * [ICUB Software]
  * 

Throughout this tutorial, we assume that all repositories are cloned into the `~\Code` directory.
**Wrapyfi should also be cloned into the `~\Code` directory in order to access the examples.**

Additionally, cloning the [Wrapyfi interfaces](https://github.com/modular-ml/wrapyfi-interfaces) repository is needed 
since it provides dedicated interfaces for communicating with the robots, acquiring and publishing webcam images, 
and providing message structures for standardizing exchanges between applications: 

```bash
cd ~/Code
git clone https://github.com/modular-ml/wrapyfi-interfaces.git
```

and add it to the `PYTHONPATH` environment variable:

```bash
export PYTHONPATH=$PYTHONPATH:~/Code/wrapyfi-interfaces
```

## Modifying the FER Model

To integrate Wrapyfi into the [facial expression recognition model](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks), we first need to modify the model to accept and return data from and to the robot interfaces.




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


### Sending the Recognized Emotion to the Robot Interfaces
[TODO]

### Distributing the Ensemble Branches
[TODO]

## Running the Application

<details>

  <summary>**Easy: iCub simulation only; running all scripts on a single machine**</summary>

  Start the yapserver to enable communication with the iCub robot:
  
  ```bash
  yarpserver
  ```

  Start the iCub simulator:
    
  ```bash
  iCub_SIM
  ```

  The facial expressions shown on the iCub's face are not enabled by default when running the iCub simulator, so we need to start the iCubFaceExpressions module to enable them:
  
  ```bash 
  simFaceExpressions
  ```

  Start the iCub emotion interface to receive the facial expressions a specific port/topic:
  
  ```bash
  emotionInterface --name /icubSim/face/emotions --context faceExpressions --from emotions.ini
  ```

  Connect the iCub simulator ports to the iCub emotion interface:
  
  ```bash
  yarp connect /face/eyelids /icubSim/face/eyelids
  yarp connect /face/image/out /icubSim/texture/face
  yarp connect /icubSim/face/emotions/out /icubSim/face/raw/in
  ```
  
  Start the iCub interface to receive the facial expressions from the application controller and activate the facial expressions on the iCub robot:
  
  ```bash 
  cd $HOME/Code/wrapyfi-interfaces
  python3 wrapyfi_interfaces/robots/icub_head/interface.py \
  --simulation --get_cam_feed \
  --control_expressions \
  --facial_expressions_port "/control_interface/facial_expressions_icub"
  ```
  
  Start the camera interface to receive images from the webcam and forward them to the application controller:

  ```bash
  cd $HOME/Code/wrapyfi/examples/applications
  python wrapyfi_interfaces/io/video/interface.py --mware yarp --cap_source "0" --fps 30 --cap_feed_port "/control_interface/image_webcam" --img_width 320 --img_height 240 --jpg
  ```
  
  Start two mirrored instances of the application controller:

  ```bash
  # The first instance is responsible for running the application workflow
  cd $HOME/Code/wrapyfi/examples/applications
  WRAPYFI_DEFAULT_COMMUNICATOR="yarp" python3 affective_signaling_multirobot.py --wrapyfi_cfg wrapyfi_configs/affective_signaling_multirobot/COMP_mainpc.yml --cam_source webcam
  ```

  ```bash
  # The second instance is responsible for running the robot (iCub) control workflow
  cd $HOME/Code/wrapyfi/examples/applications
  WRAPYFI_DEFAULT_COMMUNICATOR="yarp" python3 affective_signaling_multirobot.py --wrapyfi_cfg wrapyfi_configs/affective_signaling_multirobot/OPT_icubpc.yml --cam_source webcam
  ```
  **Note**: running two instances is not necessary if we configure a single script to handle all exchanges; however, 
  we do so to separate the application workflow from the robot control workflows. In this example, where we run a single 
  robot, the utility of such separation is not apparent. If we were to merge the workflows in the main configuration 
  `COMP_mainpc.yml` file, then we must also run the workflow for **robot A** when wanting to run **robot B** only.

  ```bash
  cd $HOME/Code/wrapyfi-examples_esr9/
  export PYTHONPATH=$HOME/Code/wrapyfi-interfaces:$PYTHONPATH
  python main_esr9.py webcam -w "/control_interface/image_esr9" -d -s 2 -b --frames 10 --max_frames 10 --video_mware yarp --facial_expressions_mware yarp --facial_expressions_port "/control_interface/facial_expressions_esr9" --face_detection 3 --img_width 320 --img_height 240 --jpg
  ```
</details>