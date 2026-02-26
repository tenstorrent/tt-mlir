#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Parse command line arguments
CHECK_ONLY=false
dockbuild="$1"
if [[ "$2" == "--check-only" ]]; then
    CHECK_ONLY=true
fi

CURRENT_TT_METAL_VERSION=$(grep 'set(TT_METAL_VERSION' third_party/CMakeLists.txt | sed 's/.*"\(.*\)".*/\1/')
echo "Current tt-metal version: $CURRENT_TT_METAL_VERSION"

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

        local target=""
        if [ -n "$3" ]; then
            target="--target $3"
        fi

        echo "Building image $image_name:$DOCKER_TAG"
        docker build \
            --progress=plain \
            $target \
            --build-arg FROM_TAG=$DOCKER_TAG \
            --build-arg TT_METAL_VERSION=$CURRENT_TT_METAL_VERSION \
            -t $image_name:$DOCKER_TAG \
            -t $image_name:latest \
            -f $dockerfile .

        echo "Pushing image $image_name:$DOCKER_TAG"
        docker push $image_name:$DOCKER_TAG
      fi
    fi
}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci
build_and_push $BASE_IRD_IMAGE_NAME .github/Dockerfile.ird base-ird
build_and_push $IRD_IMAGE_NAME .github/Dockerfile.ird ird
if [ "$dockbuild" == "all" ] || [ "$dockbuild" == "cibuildwheel" ]; then
  echo "Building cibuildwheel image"
  build_and_push $CIBW_IMAGE_NAME .github/Dockerfile.cibuildwheel toolchain-source
else
  echo "Skipping cibuildwheel image build"
fi

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
