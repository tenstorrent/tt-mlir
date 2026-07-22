# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Simulator (shadow) backend: run d2m kernels as regular Python on torch.

These tests use `import d2m_jit.sim as d2m` and require **no device** and no
SYSTEM_DESC_PATH -- the kernel bodies execute on the host, so the goldens are
matched exactly (sim is more precise than the device tile path). See
tools/d2m-jit/SIMULATOR_SPEC.md.
"""

import pytest
import torch

import d2m_jit.sim as d2m
from utils import assert_pcc


def _layout(shape, block_shape=(1, 1), grid=(1, 1), dtype=d2m.float32):
    return d2m.Layout(
        shape=shape, dtype=dtype, block_shape=list(block_shape), grid_shape=list(grid)
    )


def _blocks_per_core(shape, block_shape, grid):
    mt, nt = shape[0] // 32, shape[1] // 32
    return mt // block_shape[0] // grid[0], nt // block_shape[1] // grid[1]


# --- kernels -----------------------------------------------------------------


@d2m.kernel
def add(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a + b)


@d2m.kernel
def fused_exp_add(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(lhs, [m_off + m, n_off + n])
            y = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], x.exp() + y)


@d2m.kernel
def softmax_row(x, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            t = remote_load(x, [m_off + m, n_off + n])
            row_max = reduce_max(t, 1)
            e = exp(t - row_max)
            row_sum = reduce_sum(e, 1)
            remote_store(out, [m_off + m, n_off + n], e * recip(row_sum))


@d2m.kernel
def reduce_sum_cols(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, 0], reduce_sum(x, 1))


@d2m.kernel
def reduce_max_rows(in_t, out_t, m_blocks, n_blocks):
    n_off = core_index(1) * n_blocks
    for n in range(n_blocks):
        x = remote_load(in_t, [0, n_off + n])
        remote_store(out_t, [0, n_off + n], x.reduce_max(0))


@d2m.kernel
def center_cols(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], x - reduce_mean(x, 1))


@d2m.kernel
def matmul_kernel(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a @ b)


@d2m.kernel
def matmul_transpose_b(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    remote_store(out, [0, 0], matmul(a, b, transpose_b=True))


@d2m.kernel
def select_clamp(a, b, cond, out):
    x = remote_load(a, [0, 0])
    y = remote_load(b, [0, 0])
    c = remote_load(cond, [0, 0])
    remote_store(out, [0, 0], clamp_scalar(where(c, x.tile_bcast_row(), y), -1.0, 1.0))


# --- tests -------------------------------------------------------------------


def test_eltwise_add_grid_2x2_exact():
    lhs, rhs = torch.randn(512, 512), torch.randn(512, 512)
    lin = _layout((512, 512), grid=(8, 8))
    lout = _layout((512, 512), grid=(2, 2))
    out = d2m.empty(lout)
    mb, nb = _blocks_per_core((512, 512), (1, 1), (2, 2))
    add(d2m.to_layout(lhs, lin), d2m.to_layout(rhs, lin), out, mb, nb, grid=(2, 2))
    assert torch.allclose(lhs + rhs, out.to_host(), atol=1e-5)


def test_fused_exp_add():
    lhs, rhs = torch.randn(64, 64), torch.randn(64, 64)
    layout = _layout((64, 64), grid=(2, 2))
    out = d2m.empty(layout)
    fused_exp_add(
        d2m.to_layout(lhs, layout), d2m.to_layout(rhs, layout), out, 1, 1, grid=(2, 2)
    )
    assert_pcc(torch.exp(lhs) + rhs, out.to_host())


def test_softmax_within_tile():
    x = torch.randn(32, 32)
    layout = _layout((32, 32))
    out = d2m.empty(layout)
    softmax_row(d2m.to_layout(x, layout), out, 1, 1, grid=(1, 1))
    assert_pcc(torch.softmax(x, dim=1), out.to_host())


@pytest.mark.parametrize(
    "shape,block_shape,grid",
    [
        ((32, 32), (1, 1), (1, 1)),
        ((64, 32), (1, 1), (2, 1)),
        ((64, 64), (1, 2), (1, 1)),
    ],
)
def test_reduce_sum_cols(shape, block_shape, grid):
    layout = _layout(shape, block_shape, grid)
    out = d2m.empty(d2m.reduction_layout(layout, 1))
    x = torch.randn(shape)
    mb, nb = _blocks_per_core(shape, block_shape, grid)
    reduce_sum_cols(d2m.to_layout(x, layout), out, mb, nb, grid=grid)
    result = out.to_host()
    # Reduction is per-block; with one col-block the block spans the full row.
    bn = shape[1] // 32 // grid[1]
    expected = x.reshape(shape[0], grid[1], bn * 32).sum(-1).reshape(shape[0], grid[1])
    # Single col-block per core: each core's block covers its own column span.
    assert tuple(result.shape) == (shape[0], 1)
    if grid[1] == 1:
        assert torch.allclose(x.sum(dim=1, keepdim=True), result, atol=1e-4)


def test_reduce_max_rows():
    x = torch.randn(32, 64)
    layout = _layout((32, 64), grid=(1, 2))
    out = d2m.empty(d2m.reduction_layout(layout, 0))
    reduce_max_rows(d2m.to_layout(x, layout), out, 1, 1, grid=(1, 2))
    result = out.to_host()
    assert tuple(result.shape) == (1, 64)
    assert torch.allclose(x.max(dim=0, keepdim=True).values, result, atol=1e-4)


def test_center_cols_implicit_broadcast():
    x = torch.randn(64, 64)
    layout = _layout((64, 64), block_shape=(2, 1), grid=(1, 1))
    out = d2m.empty(layout)
    mb, nb = _blocks_per_core((64, 64), (2, 1), (1, 1))  # (1, 2)
    center_cols(d2m.to_layout(x, layout), out, mb, nb, grid=(1, 1))
    expected = x.clone()
    expected[:, 0:32] = x[:, 0:32] - x[:, 0:32].mean(1, keepdim=True)
    expected[:, 32:64] = x[:, 32:64] - x[:, 32:64].mean(1, keepdim=True)
    assert torch.allclose(expected, out.to_host(), atol=1e-5)


def test_matmul_per_shard():
    lhs, rhs = torch.randn(64, 64), torch.randn(64, 64)
    layout = _layout((64, 64), grid=(2, 2))
    out = d2m.empty(layout)  # sim empty == zeros; matmul is correct without prefill
    matmul_kernel(
        d2m.to_layout(lhs, layout), d2m.to_layout(rhs, layout), out, 1, 1, grid=(2, 2)
    )
    expected = torch.zeros(64, 64)
    for gy in range(2):
        for gx in range(2):
            ly, lx = gy * 32, gx * 32
            expected[ly : ly + 32, lx : lx + 32] = (
                lhs[ly : ly + 32, lx : lx + 32] @ rhs[ly : ly + 32, lx : lx + 32]
            )
    assert_pcc(expected, out.to_host())


def test_matmul_transpose_b():
    lhs, rhs = torch.randn(32, 32), torch.randn(32, 32)
    layout = _layout((32, 32))
    out = d2m.empty(layout)
    matmul_transpose_b(
        d2m.to_layout(lhs, layout), d2m.to_layout(rhs, layout), out, grid=(1, 1)
    )
    assert_pcc(lhs @ rhs.T, out.to_host())


def test_where_clamp_tile_bcast():
    a, b = torch.randn(32, 32), torch.randn(32, 32)
    cond = (torch.randn(32, 32) > 0).float()
    layout = _layout((32, 32))
    out = d2m.empty(layout)
    select_clamp(
        d2m.to_layout(a, layout),
        d2m.to_layout(b, layout),
        d2m.to_layout(cond, layout),
        out,
        grid=(1, 1),
    )
    br = a[0:1, :].expand(32, 32)
    expected = torch.where(cond != 0, br, b).clamp(-1.0, 1.0)
    assert torch.allclose(expected, out.to_host(), atol=1e-5)


def test_zeros_full_empty():
    layout = _layout((32, 32))
    assert torch.count_nonzero(d2m.zeros(layout).to_host()).item() == 0
    assert torch.count_nonzero(d2m.empty(layout).to_host()).item() == 0
    assert torch.allclose(d2m.full(layout, 3.0).to_host(), torch.full((32, 32), 3.0))


def test_bf16_eltwise():
    lhs = torch.randn(32, 32, dtype=torch.bfloat16)
    rhs = torch.randn(32, 32, dtype=torch.bfloat16)
    layout = _layout((32, 32), dtype=d2m.bfloat16)
    out = d2m.empty(layout)
    add(d2m.to_layout(lhs, layout), d2m.to_layout(rhs, layout), out, 1, 1, grid=(1, 1))
    result = out.to_host()
    assert result.dtype == torch.bfloat16
    assert torch.allclose((lhs + rhs).float(), result.float(), atol=0.1)


# --- views -------------------------------------------------------------------


def test_permute_is_view_and_materialise():
    t = torch.randn(64, 64)
    lt = d2m.to_layout(t, _layout((64, 64), grid=(2, 2)))
    p = d2m.permute(lt, 1, 0)
    assert p.is_view is True
    out = d2m.to_layout(p, p.layout).to_host()
    assert torch.allclose(t.T, out)


def test_view_identity_round_trip():
    t = torch.randn(64, 64)
    lt = d2m.to_layout(t, _layout((64, 64), grid=(2, 2)))
    v = d2m.view(lt, lambda d0, d1: (d0, d1))
    assert v.is_view is True
    assert torch.allclose(t, d2m.to_layout(v, v.layout).to_host())


def test_view_layout_identity_round_trip():
    t = torch.randn(64, 64)
    lt = d2m.to_layout(t, _layout((64, 64), grid=(2, 2)))
    v = d2m.view_layout(lt, lambda d0, d1, d2, d3: (d0, d1, d2, d3))
    assert v.is_view is True
    assert torch.allclose(t, d2m.to_layout(v, v.layout).to_host())


def test_to_host_on_view_raises():
    t = torch.randn(64, 64)
    lt = d2m.to_layout(t, _layout((64, 64), grid=(2, 2)))
    v = d2m.permute(lt, 1, 0)
    with pytest.raises(ValueError, match="view"):
        v.to_host()


def test_permute_rejects_torch_tensor():
    with pytest.raises(TypeError):
        d2m.permute(torch.zeros(64, 64), 1, 0)


def test_view_rejects_non_permutation():
    lt = d2m.to_layout(torch.randn(64, 64), _layout((64, 64), grid=(2, 2)))
    with pytest.raises(ValueError):
        d2m.view(lt, lambda d0, d1: (d0, 0))


# --- arg validation ----------------------------------------------------------


def test_kernel_rejects_scalar_before_tensor():
    layout = _layout((32, 32))
    a = d2m.to_layout(torch.randn(32, 32), layout)
    out = d2m.empty(layout)
    with pytest.raises(TypeError, match="must precede scalars"):
        add(a, 1, out, 1, 1, grid=(1, 1))


def test_kernel_rejects_declarative_form():
    layout = _layout((32, 32))
    a = d2m.to_layout(torch.randn(32, 32), layout)
    out = d2m.empty(layout)
    with pytest.raises(NotImplementedError):
        add(a, a, out, 1, 1, grid=(1, 1), iterator_types=["parallel", "parallel"])


# --- async / semaphores ------------------------------------------------------


@d2m.kernel
async def add_async(lhs, rhs, out, m_blocks, n_blocks):
    sem = Semaphore(0)
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            a = await a
            sem.inc(1)
            sem.wait(1, reset=0)
            remote_store(out, [m_off + m, n_off + n], a + b)


def test_async_kernel_await_and_semaphore():
    """An `async def` body (await + Semaphore no-ops) is driven to completion
    and matches the synchronous result. Semaphores are ordering-only, so they
    do not affect numerics in the functional simulator."""
    lin = _layout((64, 64), grid=(2, 2))
    lhs, rhs = torch.randn(64, 64), torch.randn(64, 64)
    out = d2m.empty(lin)
    add_async(d2m.to_layout(lhs, lin), d2m.to_layout(rhs, lin), out, 1, 1, grid=(2, 2))
    assert torch.allclose(lhs + rhs, out.to_host(), atol=1e-5)


@d2m.kernel
async def gen_kernel(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    yield x
    remote_store(out_t, [0, 0], x)


def test_async_generator_kernel_rejected():
    """`async def` + `yield` (multi-thread producer/consumer) needs an ordering
    model the simulator omits; it must fail loudly rather than silently no-op."""
    layout = _layout((32, 32))
    t = torch.randn(32, 32)
    out = d2m.empty(layout)
    with pytest.raises(NotImplementedError, match="async-generator"):
        gen_kernel(d2m.to_layout(t, layout), out, grid=(1, 1))
