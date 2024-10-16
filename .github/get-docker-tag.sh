#!/bin/bash

# Calculate hash from the following files. This hash is used to tag the docker images.
# Any change in these files will result in a new docker image build
DOCKERFILE_HASH_FILES=".github/Dockerfile.base .github/Dockerfile.ci .github/Dockerfile.ci env/CMakeLists.txt"
DOCKERFILE_HASH=$(md5sum $DOCKERFILE_HASH_FILES | md5sum | cut -d ' ' -f 1)
echo dt-$DOCKERFILE_HASH
