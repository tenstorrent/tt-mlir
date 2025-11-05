#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

cmake --build $BUILD_DIR -- tt-alchemist

# upload artifact
echo "{\"name\":\"tt-alchemist-whl-$BUILD_NAME\",\"path\":\"$BUILD_DIR/tools/tt-alchemist/csrc/dist/tt_alchemist*.whl\"}," >> $UPLOAD_LIST
