#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

cd $WORK_DIR/tools/ttnn-jit
mkdir -p dist
pip install build setuptools wheel
python3 -m build --wheel

# upload artifact
echo "{\"name\":\"ttnn-jit-whl-$BUILD_NAME\",\"path\":\"$WORK_DIR/tools/ttnn-jit/dist/ttnn_jit*.whl\"}," >> $UPLOAD_LIST
