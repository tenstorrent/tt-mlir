# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for d2m-jit tile broadcasts.

This builds a lazy d2m-jit module and prints it before pipeline execution, so
the test validates the Python DSL lowering without requiring a device run.
"""

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise


@d2m.kernel
def k_tile_bcast(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            row = tile_bcast(x, "row")
            col = x.tile_bcast_col()
            bcast_2d = x.tile_bcast_2d()
            remote_store(out_t, [m_off + m, n_off + n], row + col + bcast_2d)


L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1])
inp = d2m.empty(L)
out = d2m.empty(L)
k_tile_bcast(inp, out, 1, 1, grid=(1, 1))

builder = _Builder.get()
_emit_returns_and_finalise(builder, [out])
builder.module.operation.verify()
print(builder.module)
_Builder.reset()


# CHECK-LABEL: func.func @main
# CHECK:       d2m.generic
# CHECK:       "d2m.tile_bcast"({{.*}}) <{bcast_type = #d2m<tile_bcast_type row>}>
# CHECK:       "d2m.tile_bcast"({{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
# CHECK:       "d2m.tile_bcast"({{.*}}) <{bcast_type = #d2m<tile_bcast_type scalar>}>
# CHECK:       "d2m.tile_add"
