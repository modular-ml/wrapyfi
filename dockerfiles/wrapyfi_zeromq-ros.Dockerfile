# Use micromamba as the base image
FROM mambaorg/micromamba:1.5.6

# Load micromamba automatically in bash sessions
ENV BASH_ENV=~/.bashrc
SHELL ["/bin/bash", "-c"]

# Conditionally create a micromamba environment for YARP
RUN micromamba create -n zeromq_ros -c robostack-staging -c conda-forge ros-noetic-desktop -y && \
    micromamba run -n zeromq_ros pip install wrapyfi[headless]==0.4.32 && \
    micromamba clean --all --yes


ENV ENV_NAME=zeromq_ros
CMD ["bash"]

############################################ BUILD ############################################
# docker build --rm=true --no-cache -f wrapyfi_zeromq-ros.Dockerfile -t wrapyfi-zeromq-ros .

############################################  RUN  ############################################
# docker run --name wrapyfi_zeromq_ros --net host --rm -dit wrapyfi-zeromq-ros roscore  # starts roscore. Remove -d to see output
# docker exec -e ENV_NAME=zeromq_ros -it wrapyfi_zeromq_ros bash  # starts bash in the container

###########################################   STOP  ###########################################
# docker stop $(docker ps -q --filter ancestor=wrapyfi-zeromq-ros )  # stops all containers based on this image