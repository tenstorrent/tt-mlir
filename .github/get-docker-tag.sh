#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Calculate hash from the following files. This hash is used to tag the docker images.
# Any change in these files will result in a new docker image build
DOCKERFILE_HASH_FILES=".github/Dockerfile.base .github/Dockerfile.ci .github/Dockerfile.ird .github/Dockerfile.cibuildwheel env/CMakeLists.txt env/init_venv.sh env/build-requirements.txt env/ttnn-requirements.txt env/patches/shardy.patch env/patches/shardy_mpmd_pybinds.patch test/python/requirements.txt"
DOCKERFILE_HASH=$(sha256sum $DOCKERFILE_HASH_FILES | sha256sum | cut -d ' ' -f 1)
echo dt-$DOCKERFILE_HASH
