# Use micromamba as the base image
FROM mambaorg/micromamba:1.5.6

# Load micromamba automatically in bash sessions
ENV BASH_ENV=~/.bashrc
SHELL ["/bin/bash", "-c"]

# Conditionally create a micromamba environment for YARP
RUN micromamba create -n zeromq_ros2 -c robostack-staging -c conda-forge ros-humble-desktop -y && \
    micromamba run -n zeromq_ros2 pip install wrapyfi[headless]==0.4.32 && \
    micromamba clean --all --yes

ENV ENV_NAME=zeromq_ros2
CMD ["bash"]

############################################ BUILD ############################################
# docker build --rm=true --no-cache -f wrapyfi_zeromq-ros2.Dockerfile -t wrapyfi-zeromq-ros2 .

############################################  RUN  ############################################
# docker run --net host --name wrapyfi_zeromq_ros2 --rm -it wrapyfi-zeromq-ros2  # starts bash in the container

###########################################   STOP  ###########################################
# docker stop $(docker ps -q --filter ancestor=wrapyfi-zeromq-ros2 )  # stops all containers based on this image