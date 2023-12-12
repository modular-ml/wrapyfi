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
* The ICM-20948 attached to the Pupil Core. Readings from the IMU are filtered using the Mahony algorithm [(Mahony et al., 2008)](https://ieeexplore.ieee.org/document/4608934) running on the RP2040 with a sampling frequency of 50 Hz. Both the IMU and the Pupil Core are connected to **PC:E** running the IMU interface, and the Pupil Capture software interfacing directly with the eye tracker. Since the Pupil interface communicate directly over ZeroMQ, we choose ZeroMQ for transmitting the IMU readings as well. In order to mirror the participant's head movement, we invert the values of the roll $\phi$ and yaw $\psi$. However, there may be an offset between orientation as measured by the IMU and the orientation of the participant relative to the robot. To correct this offset, we ask the participant to look straight at the robot and use the readings from 6DRepNet to shift the readings from the IMU.

 
The application manager running on **PC:A** initializes the experiment by transmitting a trigger over ZeroMQ. **PC:E** receives the trigger, starting the video feed, which is directly transmitted to **S:4** over ROS 2. To select the most suited middleware for this task, we evaluate the transmission latency of the 6DRepNet and IMU orientation coordinates with all four middleware. Two participants conducted five trials each, performing cyclic head rotations on $\theta$, $\psi$, and $\phi$---corresponding to the x,y, and z axes---independently. 

<p align="center">
  <a id="figure-2"></a>
  <img width="460" src="https://raw.githubusercontent.com/fabawi/wrapyfi/main/assets/tutorials/orientation_warping_ex1-1.png">
  <br>
  <em>Fig 2: Orientation coordinates received from the IMU (horizontal) and 6DRepNet model (vertical). Latency between de/serialization of IMU and model coordinates is measured for the best-of-five attempts using each middleware. The diagonal lines display the dynamic time warping distances between the orientation sources.</em>
</p>

The orientations inferred from the 6DRepNet and IMU were recorded for six seconds and channeled concurrently. [**Figure 2**](#figure-2) shows the best-of-five attempts with the Euclidean distance being used as a measure of alignment when performing dynamic time warping between the two head orientation sources. YARP presents the lowest latency since it is configured to acquire the last message. Due to the difference in sampling rates between the 6DRepNet model and IMU sensor, the accumulation of messages in the ZeroMQ subscriber results in a bottleneck leading to increasing latency between transmission and acquisition. With ROS and ROS 2, we set the subscriber queue size to 600 messages, allowing the subscribers to maintain all transmitted orientation coordinates without discardment. Setting their queue sizes to one leads to behavior matching that of YARP. However, the rate of dropped messages supersedes YARP significantly, with YARP dropping approximately 2% of the messages, whereas ROS and ROS 2 exceed 11% and 9%, respectively. The lowest distance offset between the two orientation sources is achieved with ROS, however, we choose YARP due to its consistent latency and relatively synchronized transmission of coordinates compared to other middleware. 

Next, **PC:C** forwards the head and eye fixation coordinates over YARP to **PC:A**. Depending on the task at hand, head orientation coordinates arriving from the 6DRepNet model or IMU sensor are transmitted along with the fixations to the iCub robot over YARP.

