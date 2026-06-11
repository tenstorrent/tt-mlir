# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for d2m-jit d2m.spatial emission."""

import d2m_jit as d2m

from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise


@d2m.kernel
def k_add(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    remote_store(out_t, [0, 0], x + x)


@d2m.kernel
def k_mul(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    remote_store(out_t, [0, 0], x * x)


L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1])
inp = d2m.empty(L)
out_add = d2m.empty(L)
out_mul = d2m.empty(L)

d2m.spatial(
    inputs=[inp],
    outputs=[out_add, out_mul],
    grid_ranges=[((0, 0), (0, 0)), ((1, 0), (1, 0))],
    region_builders=[
        lambda: k_add(inp, out_add, grid=(1, 1)),
        lambda: k_mul(inp, out_mul, grid=(1, 1)),
    ],
)

builder = _Builder.get()
_emit_returns_and_finalise(builder, [out_add, out_mul])
print(builder.module.operation.get_asm(assume_verified=True))
_Builder.reset()


# CHECK-LABEL: func.func @main
# CHECK:       d2m.spatial
# CHECK-SAME:  grid_ranges = [#ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(1,0), (1,0)>]
# CHECK:       d2m.generic
# CHECK:       "d2m.tile_add"
# CHECK:       d2m.spatial_yield
# CHECK:       d2m.generic
# CHECK-SAME:  grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1)>
# CHECK:       "d2m.tile_mul"
# CHECK:       d2m.spatial_yield
# CHECK:       return
