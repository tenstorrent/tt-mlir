# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder


@d2m.kernel
def topk_1x2048(in_t, out_t):
    row = remote_load(in_t, [0, 0])
    values = topk_values(row, 32, True, False)
    remote_store(out_t, [0, 0], values)


input_layout = d2m.Layout(
    shape=(1, 2048),
    dtype=d2m.bfloat16,
    block_shape=[1, 64],
    grid_shape=[1, 1],
)
output_layout = d2m.Layout(
    shape=(1, 32),
    dtype=d2m.bfloat16,
    block_shape=[1, 1],
    grid_shape=[1, 1],
)

in_t = d2m.empty(input_layout)
out_t = d2m.empty(output_layout)
topk_1x2048(in_t, out_t, grid=(1, 1))

print(_Builder.get().module)
_Builder.reset()


# CHECK: #ttcore.metal_layout<logical_shape = 1x2048
# CHECK-LABEL: "func.func"() <{function_type = () -> (), sym_name = "main"}
# CHECK: tensor<1x1x1x64x!ttcore.tile<32x32, bf16>
# CHECK: "d2m.generic"
# CHECK-SAME: grid = #ttcore.grid<1x1>
# CHECK: "d2m.remote_load"
# CHECK: "d2m.topk_values"
# CHECK-SAME: k = 32
# CHECK-SAME: stable_sort = false
# CHECK: "d2m.remote_store"
