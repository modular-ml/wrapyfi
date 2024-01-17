# Wrapyfi Docker

Images of supported middleware with [Wrapyfi](https://github.com/fabawi/wrapyfi) pre-installed. All middleware frameworks are installed within micromamba environments, which are activated on running the Docker container. The images have only been tested with Docker but not with Nvidia Docker.

You can pull the Docker image which suits your needs. Some images are loaded with two middleware, and others with three. Note that ROS and ROS 2 cannot be installed withing the same environment, which is why you won't find a docker with all middleware. Since we use micromamba inside the images, all middleware could be installed within an image. Given they are in separate mamba environments, we chose to keep them in separate images as well.

## Installation

There are two ways to install a Wrapyfi Docker image. You can build any of the [Dockerfiles](https://github.com/fabawi/wrapyfi/tree/main/dockerfiles) found in the Wrapyfi repository. Alternatively, you can pull the image directly from the [modularml/wrapyfi](https://hub.docker.com/repository/docker/modularml/wrapyfi) repository on the Docker Hub.

### Docker Hub [recommended]

Pull a docker image, e.g.: 

```bash
docker pull modularml/wrapyfi:0.4.32-zeromq-yarp-ros
```

### Build from source 

If you would like to modify the images before using them, you can build the [Dockerfiles](https://github.com/fabawi/wrapyfi/tree/main/dockerfiles) found in the Wrapyfi repository e.g. to build the [wrapyfi_zeromq-ros2.Dockerfile](https://github.com/fabawi/wrapyfi/blob/dev/dockerfiles/wrapyfi_zeromq-ros2.Dockerfile),

```bash
docker build --rm=true --no-cache -f wrapyfi_zeromq-ros2.Dockerfile -t wrapyfi-zeromq-ros2 .
```

## Usage

If the image requires a server (like ROS or YARP), you need to run them from within the container. To run `roscore` e.g. in `modularml/wrapyfi:0.4.32-zeromq-yarp-ros`:

```bash
docker run --name wrapyfi__roscore --net host \
        --rm -dit modularml/wrapyfi:0.4.32-zeromq-yarp-ros bash -c "roscore"
```

   **Note**: Remove the `-d` argument i.e., replace `-dit` with `-it` to keep the container attached and view the server log (including IP information).

You would also need to run the YARP server when YARP is required and a server is not running already. Images where this is applicable include `modularml/wrapyfi:0.4.32-zeromq-yarp-ros`, `modularml/wrapyfi:0.4.32-zeromq-yarp-ros2`, and `modularml/wrapyfi:0.4.32-zeromq-yarp`:

```bash
docker exec -it -e ENV_NAME=zeromq_yarp_ros wrapyfi_zeromq_yarp_ros bash -c "yarpserver"
``` 

To access an interactive bash terminal, you can run other container/s based on the existing images. Ensure that the ROS and YARP URIs and ports are correct to use the two middleware (after running `roscore` and `yarpserver`):

```bash

# for ROS environments
docker run --name wrapyfi_zeromq_yarp_ros --net host \
        --rm -it modularml/wrapyfi:0.4.32-zeromq-yarp-ros bash -c \
        "echo 'This is an environment with ROS installed' && echo 'ROS_MASTER_URI: `$ROS_MASTER_URI`'; bash"

# for YARP environments
docker run --name wrapyfi_zeromq_yarp_ros --net host \
        --rm -it modularml/wrapyfi:0.4.32-zeromq-yarp-ros bash -c \
        "echo 'This is an environment with YARP installed' && yarp detect --write && yarp name list; bash"

```

For images that don't require a server (such as ROS 2 and ZeroMQ images), you can directly run the container after pulling the corresponding image:

```bash
docker run --net host --name wrapyfi_zeromq_ros2 \
        --rm -it modularml/wrapyfi:0.4.32-zeromq-ros2 bash
```

You can also attach to an existing container which gives you access to a linux environment with pre-installed Wrapyfi and supported middleware:

```bash
docker exec -e ENV_NAME=zeromq_yarp_ros2 -it wrapyfi_zeromq_yarp_ros2 bash
```

### Mounting volumes

When running Docker in an isoluted environment, it would be useful to access/share code and resources on your local machine with the Docker container. Accessing directories on your local machine using Docker is possible through mounting. This can be done by passing the `--mount` argument to your run command, and specifying the location of the mounted directory from the source (on local machine) to the target (within Docker container to be accessed from the environment; this can be an arbitrary location), e.g.:

```bash
docker run --net host --mount \
        type=bind,source=directory/on/local/machine,target=/directory/in/docker/container \
        --name wrapyfi_zeromq_yarp --rm -it modularml/wrapyfi:0.4.32-zeromq-yarp
``` 
