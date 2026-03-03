# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: ttnn-jit

import ttnn_jit
import ttnn
import torch

from utils import create_sharded_tile_tensor


@ttnn_jit.jit(compile_only=True)
def abs_hifi4_graph(input_tensor):
    return ttnn.abs(input_tensor)


@ttnn_jit.jit(compile_only=True)
def abs_hifi4(input_tensor):
    return ttnn.abs(input_tensor)


@ttnn_jit.jit(compile_only=True, math_fidelity=ttnn.MathFidelity.HiFi3)
def abs_hifi3(input_tensor):
    return ttnn.abs(input_tensor)


@ttnn_jit.jit(compile_only=True, math_fidelity=ttnn.MathFidelity.HiFi2)
def abs_hifi2(input_tensor):
    return ttnn.abs(input_tensor)


@ttnn_jit.jit(compile_only=True, math_fidelity=ttnn.MathFidelity.LoFi)
def abs_lofi(input_tensor):
    return ttnn.abs(input_tensor)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    input_tensor = create_sharded_tile_tensor(
        device, (128, 128), (0, 0), torch.bfloat16
    )

    # CHECK: ---- IR Dump after ttnn_to_ttmetal_pipeline ----
    # CHECK: func.func @abs_hifi4_graph
    # CHECK: "ttnn.generic"
    # CHECK-SAME: math_fidelity = hifi4
    abs_hifi4_graph(input_tensor)

    # CHECK: ---- IR Dump after ttnn_to_ttmetal_pipeline ----
    # CHECK: func.func @abs_hifi4
    # CHECK: "ttnn.generic"
    # CHECK-SAME: math_fidelity = hifi4
    abs_hifi4(input_tensor)

    # CHECK: ---- IR Dump after ttnn_to_ttmetal_pipeline ----
    # CHECK: func.func @abs_hifi3
    # CHECK: "ttnn.generic"
    # CHECK-SAME: math_fidelity = hifi3
    abs_hifi3(input_tensor)

    # CHECK: ---- IR Dump after ttnn_to_ttmetal_pipeline ----
    # CHECK: func.func @abs_hifi2
    # CHECK: "ttnn.generic"
    # CHECK-SAME: math_fidelity = hifi2
    abs_hifi2(input_tensor)

    # CHECK: ---- IR Dump after ttnn_to_ttmetal_pipeline ----
    # CHECK: func.func @abs_lofi
    # CHECK: "ttnn.generic"
    # CHECK-SAME: math_fidelity = lofi
    abs_lofi(input_tensor)
    ttnn.close_device(device)
