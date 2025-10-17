#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

export TT_METAL_HOME="${{ steps.strings.outputs.work-dir }}/third_party/tt-metal/src/tt-metal"
export TT_METAL_LIB="${{ steps.strings.outputs.install-output-dir }}/lib"
cd tools/ttnn-standalone
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build -- ttnn-standalone
cp build/ttnn-standalone $INSTALL_DIR/tools/ttnn-standalone
