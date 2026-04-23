# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for d2m-jit matmul transpose_b.

This builds a lazy d2m-jit module and prints it before pipeline execution, so
the test validates Python DSL lowering without requiring a device run.
"""

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise


@d2m.kernel
def k_matmul_transpose_b(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    c = matmul(a, b, transpose_b=True)
    remote_store(out, [0, 0], c)


lhs_layout = d2m.Layout(
    shape=(32, 64), dtype=d2m.float32, block_shape=[1, 2], grid_shape=[1, 1]
)
rhs_layout = d2m.Layout(
    shape=(96, 64), dtype=d2m.float32, block_shape=[3, 2], grid_shape=[1, 1]
)
out_layout = d2m.Layout(
    shape=(32, 96), dtype=d2m.float32, block_shape=[1, 3], grid_shape=[1, 1]
)

lhs = d2m.empty(lhs_layout)
rhs = d2m.empty(rhs_layout)
out = d2m.empty(out_layout)
k_matmul_transpose_b(lhs, rhs, out, grid=(1, 1))

builder = _Builder.get()
_emit_returns_and_finalise(builder, [out])
builder.module.operation.verify()
print(builder.module)
_Builder.reset()


# CHECK-DAG:   #[[$MAP_LHS:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK-DAG:   #[[$MAP_RHS:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
# CHECK-DAG:   #[[$MAP_OUT:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
# CHECK-LABEL: func.func @main
# CHECK:       d2m.generic
# CHECK:       linalg.generic
# CHECK-SAME:  indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_OUT]]]
# CHECK:       "d2m.tile_matmul"
# CHECK-SAME:  transpose_b = true
