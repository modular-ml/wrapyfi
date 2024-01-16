# Use micromamba as the base image
FROM mambaorg/micromamba:1.5.6

# Load micromamba automatically in bash sessions
ENV BASH_ENV=~/.bashrc
SHELL ["/bin/bash", "-c"]

# Conditionally create a micromamba environment for YARP
RUN micromamba create -n zeromq_yarp_ros2 -c robostack-staging -c conda-forge ros-humble-desktop -y && \
    micromamba run -n zeromq_yarp_ros2 micromamba install -y yarp -c robotology -c conda-forge && \
    micromamba run -n zeromq_yarp_ros2 pip install wrapyfi[headless]==0.4.32 && \
    micromamba clean --all --yes


ENV ENV_NAME=zeromq_yarp_ros2
CMD ["bash"]

############################################ BUILD ############################################
# docker build --rm=true --no-cache -f wrapyfi_zeromq-yarp-ros2.Dockerfile -t wrapyfi-zeromq-yarp-ros2 .

############################################  RUN  ############################################
# docker run --name wrapyfi_zeromq_yarp_ros2 --net host --rm -dit wrapyfi-zeromq-yarp-ros2 yarpserver  # starts yarpserver. Remove -d to see output
# docker exec -e ENV_NAME=zeromq_yarp_ros2 -it wrapyfi_zeromq_yarp_ros2 bash  # starts bash in the container

###########################################   STOP  ###########################################
# docker stop $(docker ps -q --filter ancestor=wrapyfi-zeromq-yarp-ros2 )  # stops all containers based on this image