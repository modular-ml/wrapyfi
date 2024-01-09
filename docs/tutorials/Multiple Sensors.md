# Tutorial: Switching between Sensors using the Mirroring and Channeling Schemes

<p align="center">
    <video width="630" height="300" controls autoplay><source type="video/mp4" src="https://raw.githubusercontent.com/fabawi/wrapyfi/main/assets/tutorials/vid_demo_ex1-1.mp4"></video>
</p>


[Video: https://github.com/fabawi/wrapyfi/assets/4982924/6c83ef06-1d77-482d-a75f-45ad5ed81596](https://github.com/fabawi/wrapyfi/assets/4982924/6c83ef06-1d77-482d-a75f-45ad5ed81596)

This tutorial demonstrates how to use the Wrapyfi framework to run a head pose estimation model and/or acquire head orientation from inertial measurement unit (IMU) readings to mirror the movements of an actor on the iCub robot 
in a near-real-time setting. Under the model-controlled condition, the iCub robot's movements are actuated by a vision-based head pose estimation model. Under the IMU-controlled condition, the orientation readings arrived instead from an IMU attached to a wearable eye tracker.
Switching between the sources of movement estimation can be done by **channeling** the coordinates to the robot (check out the [channeling scheme](<../usage/User%20Guide/Communication%20Schemes.md#channeling>) for more details on channeling).

## Methodology

<p align="center">
  <a id="figure-1"></a>
  <img width="460" src="https://raw.githubusercontent.com/fabawi/wrapyfi/main/assets/tutorials/wrapyfi_hri_ex1-1.png">
  <br>
  <em>Fig 1: Head orientation and gaze estimation for controlling the iCub robot under the two movement conditions.</em>
</p>

Hempel et al. [(2022)](https://ieeexplore.ieee.org/document/9897219) developed 6DRepNet, a novel end-to-end neural network for head pose estimation. The authors proposed a unique solution that leverages a 6D rotation matrix representation and a geodesic distance-based loss function.
The 6D rotation matrix utilized in their approach is highly efficient for representing the orientation of objects in three-dimensional space by encoding six parameters $\text{p}_{[1,6]}$ instead of the typical nine:

```math
\begin{align}
    \mathbf{p}_x & = [\text{p}_1, \text{p}_2, \text{p}_3]
    \quad \quad
    %
    \mathbf{p}_y = [\text{p}_4, \text{p}_5, \text{p}_6]
    \quad & \quad \quad
\end{align}
```

resulting in a rotation matrix $\mathbf{R}$:


```math
\begin{align}
    \mathbf{r}_{x} & = \frac{\mathbf{p}_{x}}{\sqrt{\sum_{l=1}^{3} \text{p}_{x,l}^2}}
    \\
    \mathbf{r}_{z} & = \frac{\mathbf{r}_{x} \times \mathbf{p}_{y}}{\sqrt{\sum_{l=1}^{3} {(\text{r}_{x,l} \times \text{p}_{y,l})}^2}}
\end{align}
```

```math
\begin{align}
    \mathbf{r}_{y} & = \frac{\mathbf{r}_{z} \times \mathbf{r}_{x}}{\sqrt{\sum_{l=1}^{3} {(\text{r}_{z,l} \times \text{r}_{x,l})}^2}}
\end{align}
```

```math
\begin{align}
    \mathbf{R} & = 
    \begin{bmatrix}
    \vert & \vert & \vert
    \\
    \mathbf{r}_x^\intercal & \mathbf{r}_y^\intercal & \mathbf{r}_z^\intercal
    \\
    \vert & \vert & \vert
    \end{bmatrix}
    \equiv
    \begin{bmatrix}
    \text{R}_{11} & \text{R}_{12} & \text{R}_{13}
    \\
    \text{R}_{21} & \text{R}_{22} & \text{R}_{23}
    \\
    \text{R}_{31} & \text{R}_{32} & \text{R}_{33}
    \end{bmatrix}
    \\
\end{align}
```

which is utilized to acquire the Euler angles following the standard order (roll $\phi$, pitch $\theta$, yaw $\psi$):

```math
\begin{align}
    \mathbf{\alpha} & = \sqrt{\text{R}_{11}^2 + \text{R}_{12}^2}
    \quad
    %
    \beta = 
    \begin{cases}
        1, & \text{if } \alpha\geq 10^{-6}\\
        0, & \text{otherwise}
    \end{cases}
    \\
    \phi_M & = (1 - \beta) \cdot atan2(\text{R}_{12}, \text{R}_{11})
    \\
    \theta_M & = (1 - \beta) \cdot atan2(\text{R}_{23}, \text{R}_{33}) + \beta \cdot atan2(-\text{R}_{32}, \text{R}_{22})
    \\
    \psi_M & = atan2(-\text{R}_{13}, \alpha)
\end{align}
```

where $\phi_M$, $\theta_M$, $\psi_M$ define head orientation when the 6DRepNet model is used as the source for controlling the iCub robot.

The gaze coordinates are inferred using the Pupil Core [(Kassner et al., 2014)](https://dl.acm.org/doi/10.1145/2638728.2641695) eye tracker worn by the participant. We attach a Waveshare 9-DOF ICM-20948 IMU to a Raspberry Pi Pico RP2040 microcontroller, mounted on the upper-left rim of the Pupil Core. The eye tracker readings are inverted on the y and z axes to mirror the eye movements of the participants. We perform a single-marker eye tracker calibration prior to conducting the experiment with each participant. The two experimental conditions involved capturing eye movements. However, the head orientation source varied between the model-controlled and IMU-controlled conditions:

* The 6DRepNet [(Hempel et al., 2022)](https://ieeexplore.ieee.org/document/9897219) model for vision-based head orientation estimation. The model is implemented in PyTorch and runs with GPU support. We execute the inference script on an NVIDIA GeForce GTX 1050 Ti (denoted by **S:4** in [**Figure 1**](#figure-1)) with 4 GB VRAM, receiving $320\times240$ pixel images over ROS 2 captured using a Logitech C920 webcam with 30 FPS. The head orientation coordinates are inferred at a rate of 20 Hz.
* The ICM-20948 attached to the Pupil Core. Readings from the IMU are filtered using the Mahony algorithm [(Mahony et al., 2008)](https://ieeexplore.ieee.org/document/4608934) running on the RP2040 with a sampling frequency of 50 Hz. Both the IMU and the Pupil Core are connected to **PC:E** running the IMU interface, and the Pupil Capture software interfacing directly with the eye tracker. Since the Pupil interface communicates directly over ZeroMQ, we chose ZeroMQ for transmitting the IMU readings as well. In order to mirror the participant's head movement, we invert the values of the roll $\phi$ and yaw $\psi$. However, there may be an offset between orientation as measured by the IMU and the orientation of the participant relative to the robot. To correct this offset, we ask the participant to look straight at the robot and use the readings from 6DRepNet to shift the readings from the IMU.

 
The application manager running on **PC:A** initializes the experiment by transmitting a trigger over ZeroMQ. **PC:E** receives the trigger, starting the video feed, which is directly transmitted to **S:4** over ROS 2. To select the most suited middleware for this task, we evaluate the transmission latency of the 6DRepNet and IMU orientation coordinates with all four middleware. Two participants conducted five trials each, performing cyclic head rotations on $\theta$, $\psi$, and $\phi$---corresponding to the x,y, and z axes---independently. 

<p align="center">
  <a id="figure-2"></a>
  <img width="460" src="https://raw.githubusercontent.com/fabawi/wrapyfi/main/assets/tutorials/orientation_warping_ex1-1.png">
  <br>
  <em>Fig 2: Orientation coordinates received from the IMU (horizontal) and 6DRepNet model (vertical). Latency between de/serialization of IMU and model coordinates is measured for the best-of-five attempts using each middleware. The diagonal lines display the dynamic time warping distances between the orientation sources.</em>
</p>

The orientations inferred from the 6DRepNet and IMU were recorded for six seconds and channeled concurrently. [**Figure 2**](#figure-2) shows the best-of-five attempts with the Euclidean distance being used as a measure of alignment when performing dynamic time warping between the two head orientation sources. YARP presents the lowest latency since it is configured to acquire the last message. Due to the difference in sampling rates between the 6DRepNet model and IMU sensor, the accumulation of messages in the ZeroMQ subscriber results in a bottleneck leading to increasing latency between transmission and acquisition. With ROS and ROS 2, we set the subscriber queue size to 600 messages, allowing the subscribers to maintain all transmitted orientation coordinates without discarding them. Setting their queue sizes to one leads to behavior matching that of YARP. However, the rate of dropped messages supersedes YARP significantly, with YARP dropping approximately 2% of the messages, whereas ROS and ROS 2 exceed 11% and 9%, respectively. The lowest distance offset between the two orientation sources is achieved with ROS, however, we choose YARP due to its consistent latency and relatively synchronized transmission of coordinates compared to other middleware. 

Next, **PC:C** transmits the head and eye fixation coordinates over YARP to **PC:A**. Depending on the task at hand, head orientation coordinates arriving from the 6DRepNet model or IMU sensor are transmitted along with the fixations to the iCub robot over YARP following the channeling scheme. When the 6DRepNet model predictions are not available for any reason (camera feed is not available, view of the actor's head is obstructed, the camera is covered, or the model is unable to estimate the orientation), the application manager automatically switches to transmit readings from the IMU instead to the iCub robot. When 6DRepNet predictions are available, the IMU readings are still received (given how channeling works) but are simply ignored.

We execute the application on five machines, depending on the configuration:
* **S:4** (*mware: ROS 2, YARP*): Running the vision-based head orientation estimation model and transmitting messages to and from the application manager.
* **PC:104** (*mware: YARP*): Running on the physical iCub robot (*only needed when running the physical iCub robot*).
* **PC:A** *includes* **PC:ICUB** (*mware: YARP, ZeroMQ*): Running the iCub robot control workflow and parts of the application manager.
* **PC:C** *includes* **PC:WEBCAM** (*mware: ROS 2, ZeroMQ*): Running the webcam interface for acquiring images from the webcam and parts of the application manager.
* **PC:E** *includes* **PC:EYETRACKER** and **PC:IMU** (*mware: YARP, ZeroMQ*): Running the eye tracker and IMU interfaces. Additionally, running the application manager and channeling messages from the vision-based head orientation estimation model and IMU to the iCub robot.

At least one of either two input method PCs (**PC:WEBCAM** and **PC:EYETRACKER**) must be running for the application to work. We note that all machine scripts can be executed on a single machine, but we distribute them across multiple machines to demonstrate the flexibility of the Wrapyfi framework. Additionally, connecting the webcam, robot (physical or simulated), IMU, eye tracker, as well as running the [Pupil Capture](https://docs.pupil-labs.com/core/software/pupil-capture/) software, application managers, the YARP server, and the ZeroMQ broker would place a significant strain on memory, I/O bus, and processing resources of a single machine.

## Modifying the Vision-Based Head Orientation Estimation Model

To integrate Wrapyfi into the [6DRepNet vision-based head orientation estimation model](https://github.com/thohemp/6DRepNet), we first need to modify the model to accept and return data from and to the iCub robot interface.

This is achieved by using [Wrapyfi interfaces](https://github.com/modular-ml/wrapyfi-interfaces) which provide minimal examples of how to design the structure of templates and common interfaces, used for large-scale and complex applications. Templates and interfaces limit the types of data that can be transmitted. We can of course decide to transmit custom objects, something that Wrapyfi was designed to enable in the first place. However, in instances where we would like multiple applications to communicate and understand the information transmitted, a common structure *must* be introduced to avoid creating specific interfaces for each new application.


### Receiving and Transmitting Images from the Webcam

[TODO] 

### Sending the Orientation Coordinates to the Robot Interface

[TODO]

## Pre-requisites:

**Note**: The following installation instructions are compatible with **Ubuntu 18-22** and are not guaranteed to work on other distributions or operating systems. All installations must take place within a dedicated virtualenv, mamba/micromamba, or conda environment.

* Install [Wrapyfi](https://wrapyfi.readthedocs.io/en/latest/readme_lnk.html#installation) with all requirements (including NumPy, OpenCV, PyYAML) on all machines (excluding **PC:104**). Throughout this tutorial, we assume that all repositories are cloned into the `$HOME\Code` directory.
**Wrapyfi should also be cloned into the `$HOME\Code` directory in order to access the examples.**:

  ```bash
  cd $HOME/Code
  git clone https://github.com/fabawi/wrapyfi.git
  cd wrapyfi
  pip install .
  pip install "numpy>1.19.5,<1.26.0" "opencv-python>=4.5.5,<4.6.5.0" "pyyaml>=5.1.1"
  ```

* Install [SciPy](https://scipy.org/install/) for performing matrix rotations (on **PC:A**):
    
  ```bash
  # could be installed in several ways, but we choose pip for simplicity
  pip install "scipy==1.9.0"
  ```

* Install [pySerial](https://pyserial.readthedocs.io/en/latest/pyserial.html#installation) for accessing the IMU's serial port (on **PC:E**):
    
  ```bash
  # could be installed in several ways, but we choose pip for simplicity
  pip install "pyserial"
  ```
  
* Install [PyTorch](https://pytorch.org/get-started/locally/) for running the  vision-based head orientation estimation model (on **S:4**):

  ```bash
  # could be installed in several ways, but we choose pip for simplicity
  pip install "torch >= 1.10.1" "torchvision >= 0.11.2"
  ```

* Install the [face detection](https://github.com/elliottzheng/face-detection) library which is required by the vision-based head orientation estimation model (on **S:4**):
  
  ```bash
  pip install git+https://github.com/elliottzheng/face-detection.git@master
  ```

* Install the [6DRepNet model with Wrapyfi](https://github.com/modular-ml/wrapyfi-examples_6DRepNet) requirements (on **S:4**):
  
  ```bash
  cd $HOME/Code
  git clone https://github.com/modular-ml/wrapyfi-examples_6DRepNet.git
  cd wrapyfi-examples_6DRepNet
  pip install .
  ```

Cloning the [Wrapyfi interfaces](https://github.com/modular-ml/wrapyfi-interfaces) repository on all machines (excluding **PC:104**) is needed 
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

### Compiling the Waveshare 9-DOF ICM-20948 IMU Filter on the Raspberry Pi Pico RP2040 Microcontroller:

[TODO] 

### Installing the Pupil Capture:

[TODO] 

### Setting Up the iCub Robot:

**Note**: Installation instructions apply to **PC:A** (**PC:ICUB**). They can also be followed for **PC:E**, and **S:4**, however, only YARP with Python bindings is needed for these machines. If these machines have their required packages and Wrapyfi installed inside a mamba or micromamba environment, then installing the following within the environment should suffice: `micromamba install -c robotology yarp`

* Install [YARP](https://yarp.it/latest//install_yarp_linux.html) and [iCub Software](https://icub-tech-iit.github.io/documentation/sw_installation/) on local system following our [recommended instructions](https://wrapyfi.readthedocs.io/en/latest/yarp_install_lnk.html)
**or**
within a mamba or micromamba environment using the [robotology-superbuild](https://github.com/robotology/robotology-superbuild/blob/master/doc/conda-forge.md): 
* Activate and source YARP ([step 5 in installing YARP](https://wrapyfi.readthedocs.io/en/latest/yarp_install_lnk.html#installing-yarp)) on local system
**or**
activate the robotology-superbuild env: `micromamba activate robotologyenv`


