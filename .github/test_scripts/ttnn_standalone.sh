#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

echo "Running ttnn-standalone"
export TT_METAL_HOME="$INSTALL_DIR/tt-metal"
export TT_METAL_LIB="$INSTALL_DIR/lib"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:${TTMLIR_TOOLCHAIN_DIR}/lib:${LD_LIBRARY_PATH}"
cd "$INSTALL_DIR/tools/ttnn-standalone"
./run
