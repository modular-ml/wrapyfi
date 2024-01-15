# Use micromamba as the base image
FROM mambaorg/micromamba:1.5.6

# Load micromamba automatically in bash sessions
ENV BASH_ENV=~/.bashrc
SHELL ["/bin/bash", "-c"]

# Conditionally create a micromamba environment for YARP
RUN micromamba create -n zeromq_yarp -c robotology -c conda-forge yarp python=3 pip -y && \
    micromamba run -n zeromq_yarp pip install wrapyfi[headless]==0.4.32 && \
    micromamba clean --all --yes

ENV ENV_NAME=zeromq_yarp
CMD ["bash"]

############################################ BUILD ############################################
# docker build --rm=true --no-cache -f wrapyfi_zeromq-yarp.Dockerfile -t wrapyfi-zeromq-yarp .

############################################  RUN  ############################################
# docker run --net host --name wrapyfi_zeromq_yarp --rm -dit wrapyfi-zeromq-yarp yarsperver  # starts yarpserver. Remove -d to see output
# docker exec -e ENV_NAME=zeromq_yarp -it wrapyfi_zeromq_yarp bash  # starts bash in the container

###########################################   STOP  ###########################################
# docker stop $(docker ps -q --filter ancestor=wrapyfi-zeromq-yarp )  # stops all containers based on this image