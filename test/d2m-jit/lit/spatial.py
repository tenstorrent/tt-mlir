# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for d2m-jit d2m.spatial emission.

Mirrors the expression axes in test/d2m-jit/test_spatial.py:
  1. multi-region nesting with different kernels / shared input
  2. non-shared inputs across regions
  3. heterogeneous kernels (eltwise + matmul) in one spatial
  4. multicore matmul + multicore eltwise in one spatial
  5. three regions in one spatial
"""

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


@d2m.kernel
def k_exp(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    remote_store(out_t, [0, 0], exp(x))


@d2m.kernel
def k_exp_multicore(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], exp(x))


@d2m.kernel
def k_matmul(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    remote_store(out, [0, 0], a @ b)


@d2m.kernel
def k_matmul_multicore(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a @ b)


def _finalize(label, outs):
    builder = _Builder.get()
    _emit_returns_and_finalise(builder, list(outs))
    print(label)
    print(builder.module.operation.get_asm(assume_verified=True))
    _Builder.reset()


# --- 1. Multi-region shared input -------------------------------------------

L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1])
inp = d2m.empty(L)
out_add = d2m.empty(L)
out_mul = d2m.empty(L)

d2m.spatial(
    inputs=[inp],
    outputs=[out_add, out_mul],
    grid_ranges=[((0, 0), (1, 1)), ((1, 0), (1, 1))],
    region_builders=[
        lambda: k_add(inp, out_add, grid=(1, 1)),
        lambda: k_mul(inp, out_mul, grid=(1, 1)),
    ],
)
_finalize("MULTI_REGION_SHARED_INPUT_IR", [out_add, out_mul])

# CHECK-LABEL: MULTI_REGION_SHARED_INPUT_IR
# CHECK:       d2m.spatial
# CHECK-SAME:  grid_ranges = [#ttcore.core_range<0x0, 1x1>, #ttcore.core_range<1x0, 1x1>]
# CHECK:       d2m.generic
# CHECK:       "d2m.tile_add"
# CHECK:       d2m.spatial_yield
# CHECK:       d2m.generic
# CHECK-SAME:  grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1)>
# CHECK:       "d2m.tile_mul"
# CHECK:       d2m.spatial_yield


# --- 2. Non-shared inputs ---------------------------------------------------

inp0 = d2m.empty(L)
inp1 = d2m.empty(L)
out_add = d2m.empty(L)
out_mul = d2m.empty(L)

d2m.spatial(
    inputs=[inp0, inp1],
    outputs=[out_add, out_mul],
    grid_ranges=[((0, 0), (1, 1)), ((1, 1), (1, 1))],
    region_builders=[
        lambda: k_add(inp0, out_add, grid=(1, 1)),
        lambda: k_mul(inp1, out_mul, grid=(1, 1)),
    ],
)
_finalize("INPUTS_NOT_SHARED_IR", [out_add, out_mul])

# CHECK-LABEL: INPUTS_NOT_SHARED_IR
# CHECK:       d2m.spatial
# CHECK-SAME:  grid_ranges = [#ttcore.core_range<0x0, 1x1>, #ttcore.core_range<1x1, 1x1>]
# CHECK:       d2m.generic
# CHECK:       "d2m.tile_add"
# CHECK:       d2m.spatial_yield
# CHECK:       d2m.generic
# CHECK-SAME:  grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>
# CHECK:       "d2m.tile_mul"
# CHECK:       d2m.spatial_yield


# --- 3. Heterogeneous kernels (eltwise + matmul) ----------------------------

inp = d2m.empty(L)
weight = d2m.empty(L)
out_exp = d2m.empty(L)
out_mm = d2m.empty(L)

d2m.spatial(
    inputs=[inp, weight],
    outputs=[out_exp, out_mm],
    grid_ranges=[((0, 0), (1, 1)), ((1, 0), (1, 1))],
    region_builders=[
        lambda: k_exp(inp, out_exp, grid=(1, 1)),
        lambda: k_matmul(inp, weight, out_mm, grid=(1, 1)),
    ],
)
_finalize("HETEROGENEOUS_KERNELS_IR", [out_exp, out_mm])

# CHECK-LABEL: HETEROGENEOUS_KERNELS_IR
# CHECK:       d2m.spatial
# CHECK-SAME:  grid_ranges = [#ttcore.core_range<0x0, 1x1>, #ttcore.core_range<1x0, 1x1>]
# CHECK:       d2m.generic
# CHECK:       "d2m.tile_exp"
# CHECK:       d2m.spatial_yield
# CHECK:       d2m.generic
# CHECK-SAME:  grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1)>
# CHECK:       "d2m.tile_matmul"
# CHECK:       d2m.spatial_yield


# --- 4. Multicore matmul + multicore eltwise --------------------------------

L2 = d2m.Layout(
    shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
)
lhs = d2m.empty(L2)
rhs = d2m.empty(L2)
inp = d2m.empty(L2)
out_mm = d2m.empty(L2)
out_exp = d2m.empty(L2)

d2m.spatial(
    inputs=[lhs, rhs, inp],
    outputs=[out_mm, out_exp],
    grid_ranges=[((0, 0), (2, 2)), ((2, 0), (2, 2))],
    region_builders=[
        lambda: k_matmul_multicore(lhs, rhs, out_mm, 1, 1, grid=(2, 2)),
        lambda: k_exp_multicore(inp, out_exp, 1, 1, grid=(2, 2)),
    ],
)
_finalize("MULTICORE_MATMUL_AND_ELTWISE_IR", [out_mm, out_exp])

# CHECK-LABEL: MULTICORE_MATMUL_AND_ELTWISE_IR
# CHECK:       d2m.spatial
# CHECK-SAME:  grid_ranges = [#ttcore.core_range<0x0, 2x2>, #ttcore.core_range<2x0, 2x2>]
# CHECK:       d2m.generic
# CHECK-SAME:  grid = #ttcore.grid<2x2>
# CHECK:       "d2m.tile_matmul"
# CHECK:       d2m.spatial_yield
# CHECK:       d2m.generic
# CHECK-SAME:  grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 2, d1), physical_to_virt_map = (d0, d1) -> (0, d0 - 2, d1)>
# CHECK:       "d2m.tile_exp"
# CHECK:       d2m.spatial_yield


# --- 5. Three regions -------------------------------------------------------

inp = d2m.empty(L)
out_add = d2m.empty(L)
out_mul = d2m.empty(L)
out_exp = d2m.empty(L)

d2m.spatial(
    inputs=[inp],
    outputs=[out_add, out_mul, out_exp],
    grid_ranges=[((0, 0), (1, 1)), ((0, 1), (1, 1)), ((1, 0), (1, 1))],
    region_builders=[
        lambda: k_add(inp, out_add, grid=(1, 1)),
        lambda: k_mul(inp, out_mul, grid=(1, 1)),
        lambda: k_exp(inp, out_exp, grid=(1, 1)),
    ],
)
_finalize("THREE_REGIONS_IR", [out_add, out_mul, out_exp])

# CHECK-LABEL: THREE_REGIONS_IR
# CHECK:       d2m.spatial
# CHECK-SAME:  grid_ranges = [#ttcore.core_range<0x0, 1x1>, #ttcore.core_range<0x1, 1x1>, #ttcore.core_range<1x0, 1x1>]
# CHECK:       d2m.generic
# CHECK:       "d2m.tile_add"
# CHECK:       d2m.spatial_yield
# CHECK:       d2m.generic
# CHECK-SAME:  grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0, d1 - 1)>
# CHECK:       "d2m.tile_mul"
# CHECK:       d2m.spatial_yield
# CHECK:       d2m.generic
# CHECK-SAME:  grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1)>
# CHECK:       "d2m.tile_exp"
# CHECK:       d2m.spatial_yield
