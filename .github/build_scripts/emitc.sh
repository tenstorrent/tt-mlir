#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

export LD_LIBRARY_PATH="${TTMLIR_TOOLCHAIN_DIR}/lib:${LD_LIBRARY_PATH}"
llvm-lit -sv $WORK_DIR/test/ttmlir/EmitC/TTNN
python $WORK_DIR/tools/ttnn-standalone/ci_compile_dylib.py
