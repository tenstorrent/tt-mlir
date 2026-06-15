# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for d2m-jit matmul transpose_b.

This stays below `to_host()` so lit can validate Python DSL lowering over
several block-level M/K/N shapes without requiring device execution.
"""

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise


@d2m.kernel
def k_matmul_transpose_b_func(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    c = matmul(a, b, transpose_b=True)
    remote_store(out, [0, 0], c)


@d2m.kernel
def k_matmul_transpose_b_method(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    c = a.matmul(b, transpose_b=True)
    remote_store(out, [0, 0], c)


def _layout(shape, block_shape, dtype=d2m.float32):
    return d2m.Layout(
        shape=shape, dtype=dtype, block_shape=list(block_shape), grid_shape=[1, 1]
    )


def _run_case(label, shape, lhs_block_shape, rhs_block_shape, out_block_shape, kernel):
    m, k, n = shape
    lhs = d2m.empty(_layout((m, k), lhs_block_shape))
    rhs = d2m.empty(_layout((n, k), rhs_block_shape))
    out = d2m.empty(_layout((m, n), out_block_shape))
    kernel(lhs, rhs, out, grid=(1, 1))

    builder = _Builder.get()
    _emit_returns_and_finalise(builder, [out])
    builder.module.operation.verify()
    print(label)
    print(builder.module)
    _Builder.reset()


_run_case(
    "SINGLE_TILE_TRANSPOSE_B_IR",
    (32, 32, 32),
    (1, 1),
    (1, 1),
    (1, 1),
    k_matmul_transpose_b_func,
)
_run_case(
    "WIDE_N_TRANSPOSE_B_IR",
    (32, 32, 64),
    (1, 1),
    (2, 1),
    (1, 2),
    k_matmul_transpose_b_func,
)
_run_case(
    "MULTI_K_RECTANGULAR_TRANSPOSE_B_IR",
    (32, 64, 96),
    (1, 2),
    (3, 2),
    (1, 3),
    k_matmul_transpose_b_func,
)
_run_case(
    "METHOD_FORM_TALL_MULTI_K_TRANSPOSE_B_IR",
    (64, 96, 32),
    (2, 3),
    (1, 3),
    (2, 1),
    k_matmul_transpose_b_method,
)
print("PASS matmul transpose_b")


# CHECK-LABEL: SINGLE_TILE_TRANSPOSE_B_IR
# CHECK:       logical_shape = 32x32
# CHECK:       affine_map<(d0, d1, d2) -> (d1, d2)>
# CHECK:       tensor<1x1x!ttcore.tile<32x32, f32>>
# CHECK:       "d2m.tile_matmul"

# CHECK-LABEL: WIDE_N_TRANSPOSE_B_IR
# CHECK-DAG:   logical_shape = 32x32
# CHECK-DAG:   logical_shape = 64x32
# CHECK-DAG:   logical_shape = 32x64
# CHECK:       affine_map<(d0, d1, d2) -> (d1, d2)>
# CHECK:       tensor<2x1x!ttcore.tile<32x32, f32>>
# CHECK:       tensor<1x2x!ttcore.tile<32x32, f32>>
# CHECK:       "d2m.tile_matmul"

# CHECK-LABEL: MULTI_K_RECTANGULAR_TRANSPOSE_B_IR
# CHECK-DAG:   logical_shape = 32x64
# CHECK-DAG:   logical_shape = 96x64
# CHECK-DAG:   logical_shape = 32x96
# CHECK:       affine_map<(d0, d1, d2) -> (d1, d2)>
# CHECK:       tensor<1x2x!ttcore.tile<32x32, f32>>
# CHECK:       tensor<3x2x!ttcore.tile<32x32, f32>>
# CHECK:       tensor<1x3x!ttcore.tile<32x32, f32>>
# CHECK:       "d2m.tile_matmul"

# CHECK-LABEL: METHOD_FORM_TALL_MULTI_K_TRANSPOSE_B_IR
# CHECK-DAG:   logical_shape = 64x96
# CHECK-DAG:   logical_shape = 32x96
# CHECK-DAG:   logical_shape = 64x32
# CHECK:       affine_map<(d0, d1, d2) -> (d1, d2)>
# CHECK:       tensor<2x3x!ttcore.tile<32x32, f32>>
# CHECK:       tensor<1x3x!ttcore.tile<32x32, f32>>
# CHECK:       tensor<2x1x!ttcore.tile<32x32, f32>>
# CHECK:       "d2m.tile_matmul"
# CHECK:       PASS matmul transpose_b
