#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# By default build the release configuration

# Check the first argument
if [ -z "$1" ]; then
    CONFIG="release"
elif [[ "$1" == "debug" ]]; then
    CONFIG="debug"
else
    echo "Error: Invalid argument. Use 'debug' or leave empty for default build."
    exit 1
fi

REPO=tenstorrent/tt-mlir
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-ci-ubuntu-22-04
IRD_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-ird-ubuntu-22-04
if [ "$CONFIG" = "debug" ]; then
    IRD_IMAGE_NAME=$IRD_IMAGE_NAME-debug
    CI_IMAGE_NAME=$CI_IMAGE_NAME-debug
fi

echo "Building images for config: $CONFIG"

# Compute the hash of the Dockerfile
DOCKER_TAG=$(./.github/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

# Are we on main branch
ON_MAIN=$(git branch --show-current | grep -q main && echo "true" || echo "false")

build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local on_main=$3


    if docker manifest inspect $image_name:$DOCKER_TAG > /dev/null; then
        echo "Image $image_name:$DOCKER_TAG already exists"
    else
        echo "Building image $image_name:$DOCKER_TAG"
        docker build \
            --progress=plain \
            --build-arg FROM_TAG=$DOCKER_TAG \
            --build-arg CONFIG=$CONFIG \
            -t $image_name:$DOCKER_TAG \
            -t $image_name:latest \
            -f $dockerfile .

        echo "Pushing image $image_name:$DOCKER_TAG"
        docker push $image_name:$DOCKER_TAG

        # If we are on main branch also push the latest tag
        if [ "$on_main" = "true" ]; then
            echo "Pushing image $image_name:latest"
            docker push $image_name:latest
        fi
    fi
}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base $ON_MAIN
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci $ON_MAIN
build_and_push $IRD_IMAGE_NAME .github/Dockerfile.ird $ON_MAIN

echo "All images built and pushed successfully"
