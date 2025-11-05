#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Parse command line arguments
CHECK_ONLY=false
if [[ "$1" == "--check-only" ]]; then
    CHECK_ONLY=true
fi

REPO=tenstorrent/tt-mlir
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-ci-ubuntu-22-04
BASE_IRD_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-base-ird-ubuntu-22-04
IRD_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-ird-ubuntu-22-04
CIBW_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-manylinux-2-34

# Compute the hash of the Dockerfile
DOCKER_TAG=$(./.github/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local from_image=$3

    IMAGE_EXISTS=false
    if docker manifest inspect $image_name:$DOCKER_TAG > /dev/null; then
        IMAGE_EXISTS=true
    fi

    if [ "$IMAGE_EXISTS" = true ]; then
        echo "Image $image_name:$DOCKER_TAG already exists"
        if [ "$CHECK_ONLY" = true ]; then
          return 0
        fi
    else
      if [ "$CHECK_ONLY" = true ]; then
        echo "Image $image_name:$DOCKER_TAG does not exist (check-only mode)"
        return 2
      else
        echo "Docker build neccessary, ensure dependencies for toolchain build..."
        sudo apt-get update && sudo apt-get install -y cmake build-essential

        echo "Building image $image_name:$DOCKER_TAG"
        docker build \
            --progress=plain \
            --build-arg FROM_TAG=$DOCKER_TAG \
            ${from_image:+--build-arg FROM_IMAGE=$from_image} \
            -t $image_name:$DOCKER_TAG \
            -t $image_name:latest \
            -f $dockerfile .

        echo "Pushing image $image_name:$DOCKER_TAG"
        docker push $image_name:$DOCKER_TAG
      fi
    fi
}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base
build_and_push $BASE_IRD_IMAGE_NAME .github/Dockerfile.ird base
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci
build_and_push $IRD_IMAGE_NAME .github/Dockerfile.ird ci
build_and_push $CIBW_IMAGE_NAME .github/Dockerfile.cibuildwheel

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
