Images of supported middleware with [Wrapyfi](https://github.com/fabawi/wrapyfi) pre-installed. All middleware frameworks are installed within micromamba environments, which are activated on running the Docker container. The images have only been tested with Docker but not with Nvidia Docker.

You can pull the Docker image which suits your needs. Some images are loaded with two middleware, and others with three. Note that ROS and ROS 2 cannot be installed withing the same environment, which is why you won't find a docker with all middleware. Since we use micromamba inside the images, all middleware could be installed within an image. Given they are in separate mamba environments, we chose to keep them in separate images as well.

## Installation

There are two ways to install a Wrapyfi Docker image. You can build any of the [Dockerfiles](https://github.com/fabawi/wrapyfi/tree/main/dockerfiles/README.md) found in the Wrapyfi repository. Alternatively, you can pull the image directly from the [modularml/wrapyfi](https://hub.docker.com/repository/docker/modularml/wrapyfi) repository on the Docker Hub.

### Docker Hub [recommended]

Pull a docker image, e.g.: 

```bash
docker pull modularml/wrapyfi:0.4.32-zeromq-yarp-ros
```

### Build from source 

If you would like to modify the images before using them, you can build the [Dockerfiles](https://github.com/fabawi/wrapyfi/tree/main/dockerfiles/README.md) found in the Wrapyfi repository e.g. to build the [wrapyfi_zeromq-ros2.Dockerfile](https://github.com/fabawi/wrapyfi/blob/dev/dockerfiles/wrapyfi_zeromq-ros2.Dockerfile),

```bash
docker build --rm=true --no-cache -f wrapyfi_zeromq-ros2.Dockerfile -t wrapyfi-zeromq-ros2 .
```

## Usage

If the image requires a server (like ROS or YARP), you need to run them from within the container, e.g.:

```bash
docker run --name wrapyfi_zeromq_yarp_ros --net host \
        --rm -dit docker pull modularml/wrapyfi:0.4.32-zeromq-yarp-ros roscore
```

   **Note**: Remove the `-d` argument i.e., replace `-dit` with `-it` to keep the container attached and view the server log. 

You would also need to run the YARP server in `modularml/wrapyfi:0.4.32-zeromq-yarp-ros` . But since the container is already running, you can `exec` the command---However, you cannot detach it:

```bash
docker exec -e ENV_NAME=zeromq_yarp_ros wrapyfi_zeromq_yarp_ros bash & yarpserver
``` 

Now you can attach to the container which gives you access to a linux environment with pre-installed Wrapyfi and supported middleware:

```bash
docker exec -e ENV_NAME=zeromq_yarp_ros -it wrapyfi_zeromq_yarp_ros bash
``` 

For images that don't require a server (such as ROS 2 and ZeroMQ images), you simply run the container after pulling the corresponding image:

```bash
docker run --net host --name wrapyfi_zeromq_ros2 \
        --rm -it modularml/wrapyfi:0.4.32-zeromq-ros2
```

### Mounting volumes

When running Docker in an isoluted environment, it would be useful to access/share code and resources on your local machine with the Docker container. Accessing directories on your local machine using Docker is possible through mounting. This can be done by passing the `--mount` argument to your run command, and specifying the location of the mounted directory from the source (on local machine) to the target (within Docker container to be accessed from the environment; this can be an arbitrary location), e.g.:

```bash
docker run --net host --mount \
        type=bind,source=directory/on/local/machine,target=/directory/in/docker/container \
        --name wrapyfi_zeromq_yarp --rm -it modularml/wrapyfi:0.4.32-zeromq-yarp
``` 
