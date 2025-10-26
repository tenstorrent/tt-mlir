#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

export TT_METAL_RUNTIME_ROOT="$INSTALL_DIR/tt-metal"
export TT_METAL_LIB="$INSTALL_DIR/lib"
cd tools/ttnn-standalone
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build -- ttnn-standalone
cp build/ttnn-standalone $INSTALL_DIR/tools/ttnn-standalone
