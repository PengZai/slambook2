#!/bin/bash

PROJECT_NAME="slambook2"
IMAGE_NAME="${PROJECT_NAME}:ubuntu18.04"
DATA_PATH="/media/${USER}/zhipeng_usb/datasets"
DATA_PATH2="/media/${USER}/zhipeng_8t/datasets"
# Pick up config image key if specified
if [[ ! -z "${CONFIG_DATA_PATH}" ]]; then
    DATA_PATH=$CONFIG_DATA_PATH
fi



docker build -t $IMAGE_NAME -f "${HOME}/vscode_projects/${PROJECT_NAME}/Docker/Dockerfile" .


xhost +local:root

docker run \
    -e DISPLAY=$DISPLAY \
    -v ~/.Xauthority:/root/.Xauthority:rw \
    --network host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${HOME}/vscode_projects/${PROJECT_NAME}:/root/${PROJECT_NAME} \
    -v ${DATA_PATH}:/root/datasets \
    -v ${DATA_PATH2}:/root/datasets2 \
    --privileged \
    --cap-add sys_ptrace \
    --runtime=nvidia \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    -it --name $PROJECT_NAME $IMAGE_NAME /bin/bash

# docker run --rm -it --name $PROJECT_NAME $IMAGE_NAME /bin/bash

# xhost -local:root