# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Probe d2m-jit authoring limits for `d2m.spatial`.

Each test covers one expression axis (not spatial-op functional coverage):
  1. multi-region nesting with different kernels / shared input
  2. non-shared inputs across regions
  3. heterogeneous kernels (eltwise + matmul) in one spatial
  4. multicore matmul + multicore eltwise in one spatial
  5. three regions in one spatial
"""

import torch
import d2m_jit as d2m


@d2m.kernel
def k_spatial_add(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    remote_store(out_t, [0, 0], x + x)


@d2m.kernel
def k_spatial_mul(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    remote_store(out_t, [0, 0], x * x)


@d2m.kernel
def k_spatial_exp(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    remote_store(out_t, [0, 0], exp(x))


@d2m.kernel
def k_spatial_exp_multicore(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], exp(x))


@d2m.kernel
def k_spatial_matmul(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    remote_store(out, [0, 0], a @ b)


@d2m.kernel
def k_spatial_matmul_multicore(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a @ b)


def _per_shard_matmul(lhs, rhs, grid_y=2, grid_x=2, tile=32):
    expected = torch.zeros_like(lhs)
    for gy in range(grid_y):
        for gx in range(grid_x):
            ly, lx = gy * tile, gx * tile
            expected[ly : ly + tile, lx : lx + tile] = (
                lhs[ly : ly + tile, lx : lx + tile]
                @ rhs[ly : ly + tile, lx : lx + tile]
            )
    return expected


def test_spatial_two_regions_two_outputs():
    """Multi-region nesting: different kernels, shared input, offset region."""
    t = torch.randn(32, 32, dtype=torch.float32)
    in_layout = d2m.Layout(
        shape=t.shape, dtype=d2m.float32, block_shape=[1, 1], mem_space="dram"
    )
    out_layout = d2m.Layout(shape=t.shape, dtype=d2m.float32, block_shape=[1, 1])

    inp = d2m.to_layout(t, in_layout)
    out_add = d2m.empty(out_layout)
    out_mul = d2m.empty(out_layout)

    d2m.spatial(
        inputs=[inp],
        outputs=[out_add, out_mul],
        grid_ranges=[((0, 0), (0, 0)), ((1, 0), (1, 0))],
        region_builders=[
            lambda: k_spatial_add(inp, out_add, grid=(1, 1)),
            lambda: k_spatial_mul(inp, out_mul, grid=(1, 1)),
        ],
    )

    add_result, mul_result = d2m.to_host(out_add, out_mul)
    add_diff = (t + t - add_result).abs().max().item()
    mul_diff = (t * t - mul_result).abs().max().item()
    assert add_diff < 0.01, f"spatial add max diff {add_diff}"
    assert mul_diff < 0.05, f"spatial mul max diff {mul_diff}"


def test_spatial_inputs_not_shared():
    """Non-shared inputs: each region references a distinct input tensor."""
    t0 = torch.randn(32, 32, dtype=torch.float32)
    t1 = torch.randn(32, 32, dtype=torch.float32)
    in_layout = d2m.Layout(
        shape=t0.shape, dtype=d2m.float32, block_shape=[1, 1], mem_space="dram"
    )
    out_layout = d2m.Layout(shape=t0.shape, dtype=d2m.float32, block_shape=[1, 1])

    inp0 = d2m.to_layout(t0, in_layout)
    inp1 = d2m.to_layout(t1, in_layout)
    out_add = d2m.empty(out_layout)
    out_mul = d2m.empty(out_layout)

    d2m.spatial(
        inputs=[inp0, inp1],
        outputs=[out_add, out_mul],
        grid_ranges=[((0, 0), (0, 0)), ((1, 1), (1, 1))],
        region_builders=[
            lambda: k_spatial_add(inp0, out_add, grid=(1, 1)),
            lambda: k_spatial_mul(inp1, out_mul, grid=(1, 1)),
        ],
    )

    add_result, mul_result = d2m.to_host(out_add, out_mul)
    add_diff = (t0 + t0 - add_result).abs().max().item()
    mul_diff = (t1 * t1 - mul_result).abs().max().item()
    assert add_diff < 0.01, f"spatial non-shared add max diff {add_diff}"
    assert mul_diff < 0.05, f"spatial non-shared mul max diff {mul_diff}"


def test_spatial_heterogeneous_kernels():
    """Heterogeneous kernels: eltwise and matmul regions in one spatial."""
    t = torch.randn(32, 32, dtype=torch.float32) * 0.25
    w = torch.randn(32, 32, dtype=torch.float32) * 0.25
    in_layout = d2m.Layout(
        shape=t.shape, dtype=d2m.float32, block_shape=[1, 1], mem_space="dram"
    )
    out_layout = d2m.Layout(shape=t.shape, dtype=d2m.float32, block_shape=[1, 1])

    inp = d2m.to_layout(t, in_layout)
    weight = d2m.to_layout(w, in_layout)
    out_exp = d2m.empty(out_layout)
    out_mm = d2m.empty(out_layout)

    d2m.spatial(
        inputs=[inp, weight],
        outputs=[out_exp, out_mm],
        grid_ranges=[((0, 0), (0, 0)), ((1, 0), (1, 0))],
        region_builders=[
            lambda: k_spatial_exp(inp, out_exp, grid=(1, 1)),
            lambda: k_spatial_matmul(inp, weight, out_mm, grid=(1, 1)),
        ],
    )

    exp_result, mm_result = d2m.to_host(out_exp, out_mm)
    exp_diff = (torch.exp(t) - exp_result).abs().max().item()
    mm_diff = (t @ w - mm_result).abs().max().item()
    assert exp_diff < 0.05, f"spatial heterogeneous exp max diff {exp_diff}"
    assert mm_diff < 0.1, f"spatial heterogeneous matmul max diff {mm_diff}"


def test_spatial_multicore_matmul_and_eltwise():
    """Multicore matmul + multicore eltwise regions in one spatial."""
    lhs = torch.randn(64, 64, dtype=torch.float32) * 0.25
    rhs = torch.randn(64, 64, dtype=torch.float32) * 0.25
    t = torch.randn(64, 64, dtype=torch.float32) * 0.5
    layout = d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
        mem_space="dram",
    )
    out_layout = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )

    lhs_d = d2m.to_layout(lhs, layout)
    rhs_d = d2m.to_layout(rhs, layout)
    inp = d2m.to_layout(t, layout)
    out_mm = d2m.empty(out_layout)
    out_exp = d2m.empty(out_layout)

    d2m.spatial(
        inputs=[lhs_d, rhs_d, inp],
        outputs=[out_mm, out_exp],
        grid_ranges=[((0, 0), (1, 1)), ((2, 0), (3, 1))],
        region_builders=[
            lambda: k_spatial_matmul_multicore(lhs_d, rhs_d, out_mm, 1, 1, grid=(2, 2)),
            lambda: k_spatial_exp_multicore(inp, out_exp, 1, 1, grid=(2, 2)),
        ],
    )

    mm_result, exp_result = d2m.to_host(out_mm, out_exp)
    mm_diff = (_per_shard_matmul(lhs, rhs) - mm_result).abs().max().item()
    exp_diff = (torch.exp(t) - exp_result).abs().max().item()
    assert mm_diff < 0.1, f"spatial multicore matmul max diff {mm_diff}"
    assert exp_diff < 0.05, f"spatial multicore eltwise max diff {exp_diff}"


def test_spatial_three_regions():
    """Three regions in one spatial: add, mul, and exp on distinct cores."""
    t = torch.randn(32, 32, dtype=torch.float32) * 0.5
    in_layout = d2m.Layout(
        shape=t.shape, dtype=d2m.float32, block_shape=[1, 1], mem_space="dram"
    )
    out_layout = d2m.Layout(shape=t.shape, dtype=d2m.float32, block_shape=[1, 1])

    inp = d2m.to_layout(t, in_layout)
    out_add = d2m.empty(out_layout)
    out_mul = d2m.empty(out_layout)
    out_exp = d2m.empty(out_layout)

    d2m.spatial(
        inputs=[inp],
        outputs=[out_add, out_mul, out_exp],
        grid_ranges=[((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 0))],
        region_builders=[
            lambda: k_spatial_add(inp, out_add, grid=(1, 1)),
            lambda: k_spatial_mul(inp, out_mul, grid=(1, 1)),
            lambda: k_spatial_exp(inp, out_exp, grid=(1, 1)),
        ],
    )

    add_result, mul_result, exp_result = d2m.to_host(out_add, out_mul, out_exp)
    add_diff = (t + t - add_result).abs().max().item()
    mul_diff = (t * t - mul_result).abs().max().item()
    exp_diff = (torch.exp(t) - exp_result).abs().max().item()
    assert add_diff < 0.01, f"spatial three-region add max diff {add_diff}"
    assert mul_diff < 0.05, f"spatial three-region mul max diff {mul_diff}"
    assert exp_diff < 0.05, f"spatial three-region exp max diff {exp_diff}"
