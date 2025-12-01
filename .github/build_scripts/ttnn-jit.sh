#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

cd $WORK_DIR/tools/ttnn-jit
rm -rf build
pip install build setuptools wheel
TTMLIR_DEV_BUILD=ON python -m build --wheel --outdir build

# upload artifact
echo "{\"name\":\"ttnn-jit-whl-$BUILD_NAME\",\"path\":\"$WORK_DIR/tools/ttnn-jit/build/ttnn_jit*.whl\"}," >> $UPLOAD_LIST
