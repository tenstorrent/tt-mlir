#!/bin/bash
set -e

REPO=tenstorrent/tt-mlir
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-ci-ubuntu-22-04
BASE_IRD_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-base-ird-ubuntu-22-04
IRD_IMAGE_NAME=ghcr.io/$REPO/tt-mlir-ird-ubuntu-22-04

# Compute the hash of the Dockerfile
DOCKER_TAG=$(./.github/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

# Get the SHA of the current commit
CURRENT_SHA=$(git rev-parse HEAD)
echo "CURRENT_SHA: $CURRENT_SHA"

build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local from_image=$3

    if docker manifest inspect $image_name:$DOCKER_TAG; then
        echo "Image $image_name:$DOCKER_TAG already exists"
    else
        echo "Building image $image_name:$DOCKER_TAG"
        echo docker build \
            --build-arg GIT_SHA=$CURRENT_SHA \
            ${from_image:+--build-arg FROM_IMAGE=$from_image} \
            -t $image_name:$DOCKER_TAG \
            -t $image_name:latest \
            -f $dockerfile .
        echo "Pushing image $image_name:$DOCKER_TAG"
        echo docker push $image_name:$DOCKER_TAG
        echo docker push $image_name:latest
    fi
}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base
build_and_push $BASE_IRD_IMAGE_NAME .github/Dockerfile.ird base
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci
build_and_push $IRD_IMAGE_NAME .github/Dockerfile.ird ci

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG

