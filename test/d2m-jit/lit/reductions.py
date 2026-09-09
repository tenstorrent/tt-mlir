# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for d2m-jit float reductions.

This stays below `to_host()` so lit can validate the builder contract without
requiring device execution.
"""

import torch
import d2m_jit as d2m

from d2m_jit._src.builder import _Builder

_L = d2m.Layout(
    shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
)
_L_WIDE_BLOCK = d2m.Layout(
    shape=(32, 64), dtype=d2m.float32, block_shape=[1, 2], grid_shape=[1, 1]
)
_L_REDUCE_ROWS_256 = d2m.Layout(
    shape=(256, 256), dtype=d2m.float32, block_shape=[8, 1], grid_shape=[1, 8]
)
_L_REDUCE_COLS_256 = d2m.Layout(
    shape=(256, 256), dtype=d2m.float32, block_shape=[1, 8], grid_shape=[8, 1]
)


def test_basic_reduction_ir_shape():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        s = reduce_sum(x, 1)
        m = x.reduce_max(0)
        a = reduce_mean(x, -1)
        y = s.add(m).add(a)
        remote_store(out_t, [0, 0], y)

    t = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    k(d2m.to_layout(t, _L), d2m.empty(_L), grid=(1, 1))
    print("BASIC_REDUCTION_IR")
    print(_Builder.get().module)
    _Builder.reset()


def test_reduced_output_ir_shape():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        remote_store(out_t, [0, 0], reduce_sum(x, 1))

    t = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    k(d2m.to_layout(t, _L), d2m.empty(d2m.reduction_layout(_L, 1)), grid=(1, 1))
    print("REDUCED_OUTPUT_IR")
    print(_Builder.get().module)
    _Builder.reset()


def test_multi_tile_single_core_reduction_ir_shape():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        remote_store(out_t, [0, 0], x.reduce_mean(1))

    t = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    k(
        d2m.to_layout(t, _L_WIDE_BLOCK),
        d2m.empty(d2m.reduction_layout(_L_WIDE_BLOCK, 1)),
        grid=(1, 1),
    )
    print("MULTI_TILE_SINGLE_CORE_REDUCTION_IR")
    print(_Builder.get().module)
    _Builder.reset()


def test_multi_core_dim0_reduction_ir_shape():
    @d2m.kernel
    def k(in_t, out_t):
        n = core_index(1)
        x = remote_load(in_t, [0, n])
        remote_store(out_t, [0, n], reduce_sum(x, 0))

    t = torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256)
    k(
        d2m.to_layout(t, _L_REDUCE_ROWS_256),
        d2m.empty(d2m.reduction_layout(_L_REDUCE_ROWS_256, 0)),
        grid=(1, 8),
    )
    print("MULTI_CORE_DIM0_REDUCTION_IR")
    print(_Builder.get().module)
    _Builder.reset()


def test_multi_core_dim1_reduction_ir_shape():
    @d2m.kernel
    def k(in_t, out_t):
        m = core_index(0)
        x = remote_load(in_t, [m, 0])
        remote_store(out_t, [m, 0], reduce_sum(x, 1))

    t = torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256)
    k(
        d2m.to_layout(t, _L_REDUCE_COLS_256),
        d2m.empty(d2m.reduction_layout(_L_REDUCE_COLS_256, 1)),
        grid=(8, 1),
    )
    print("MULTI_CORE_DIM1_REDUCTION_IR")
    print(_Builder.get().module)
    _Builder.reset()


test_basic_reduction_ir_shape()
test_reduced_output_ir_shape()
test_multi_tile_single_core_reduction_ir_shape()
test_multi_core_dim0_reduction_ir_shape()
test_multi_core_dim1_reduction_ir_shape()
print("PASS reductions")


# CHECK-LABEL: BASIC_REDUCTION_IR
# CHECK: "func.func"
# CHECK: d2m.to_layout
# CHECK: "d2m.generic"
# CHECK: d2m.remote_load
# CHECK: d2m.tile_reduce_sum
# CHECK-SAME: reduce_dim = #d2m<reduce_dim R>
# CHECK: d2m.tile_reduce_max
# CHECK-SAME: reduce_dim = #d2m<reduce_dim C>
# CHECK: d2m.tile_reduce_mean
# CHECK-SAME: reduce_dim = #d2m<reduce_dim R>
# CHECK: d2m.tile_bcast

# CHECK-LABEL: REDUCED_OUTPUT_IR
# CHECK: logical_shape = 32x1
# CHECK: d2m.tile_reduce_sum
# CHECK-SAME: reduce_dim = #d2m<reduce_dim R>

# CHECK-LABEL: MULTI_TILE_SINGLE_CORE_REDUCTION_IR
# CHECK: logical_shape = 32x64
# CHECK: logical_shape = 32x1
# CHECK: affine_map<(d0, d1) -> (d0, 0)>
# CHECK: affine_map<(d0, d1) -> (d0, 1)>
# CHECK: tensor<1x2x!ttcore.tile<32x32, f32>>
# CHECK: tensor<1x1x!ttcore.tile<32x32, f32>>
# CHECK: d2m.tile_reduce_mean
# CHECK-SAME: reduce_dim = #d2m<reduce_dim R>
# CHECK: d2m.tile_reduce_mean
# CHECK-SAME: reduce_dim = #d2m<reduce_dim R>

# CHECK-LABEL: MULTI_CORE_DIM0_REDUCTION_IR
# CHECK: logical_shape = 256x256
# CHECK: logical_shape = 1x256
# CHECK: tensor<8x1x!ttcore.tile<32x32, f32>>
# CHECK: tensor<1x1x!ttcore.tile<32x32, f32>>
# CHECK: d2m.tile_reduce_sum
# CHECK-SAME: reduce_dim = #d2m<reduce_dim C>

# CHECK-LABEL: MULTI_CORE_DIM1_REDUCTION_IR
# CHECK: logical_shape = 256x256
# CHECK: logical_shape = 256x1
# CHECK: tensor<1x8x!ttcore.tile<32x32, f32>>
# CHECK: tensor<1x1x!ttcore.tile<32x32, f32>>
# CHECK: d2m.tile_reduce_sum
# CHECK-SAME: reduce_dim = #d2m<reduce_dim R>
# CHECK: PASS reductions
