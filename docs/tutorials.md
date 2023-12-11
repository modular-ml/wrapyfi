
We provide a set of tutorials to demonstrate the use of the Wrapyfi framework for distributed machine learning and robotics applications.

# Robotics

## [Multiple Robot Control using the Mirroring and Forwarding Schemes](./tutorials/Multiple%20Robots.md)
This tutorial demonstrates how to use the Wrapyfi framework to run a facial expression recognition (FER) model on multiple robots. 
The model recognizes 8 facial expressions which are propagated to the Pepper and iCub robots. The expression categories are displayed by changing the Pepper robot's eye and shoulder LED colors---or 
\textit{robotic facial expressions}---by changing the iCub robot's eyebrow and mouth LED patterns. 

## [Switching between Sensors using the Mirroring and Channeling Schemes](./tutorials/Multiple%20Sensors.md)
This tutorial demonstrates how to use the Wrapyfi framework to run a head pose estimation model and/or acquire head orientation from inertial measurement unit (IMU) readings to mirror the movements of an actor on the iCub robot in a near-real-time setting. Under the model-controlled condition, the iCub robot's movements are actuated by a vision-based head pose estimation model. Under the IMU-controlled condition, the orientation readings arrived instead from an IMU attached to a wearable eye tracker. 

# Neural Networks

## [Horizontal and Vertical Layer Sharding](./tutorials/Layer%20Sharding.md)
This tutorial demonstrates how to use the Wrapyfi to shard a neural network across multiple machines. We provide two examples: (1) horizontal sharding of the facial expression recognition model, and (2) vertical sharding of the Llama LLM model.