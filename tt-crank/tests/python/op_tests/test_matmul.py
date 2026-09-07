# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for aten::matmul (N-D) and aten::bmm."""

import pytest
import torch

from tt_kurbla.torch.testing import assert_close_cpu_vs_tt


@pytest.mark.parametrize(
    "m,k,n",
    [(32, 64, 32), (64, 128, 64), (32, 32, 32)],
)
def test_matmul_2d(m: int, k: int, n: int) -> None:
    a = torch.randn((m, k), dtype=torch.bfloat16)
    b = torch.randn((k, n), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.matmul, a, b, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "batch,m,k,n",
    [(2, 32, 64, 32), (4, 64, 32, 64)],
)
def test_matmul_3d(batch: int, m: int, k: int, n: int) -> None:
    a = torch.randn((batch, m, k), dtype=torch.bfloat16)
    b = torch.randn((batch, k, n), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.matmul, a, b, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "b,h,s,d",
    [(1, 8, 128, 64), (1, 32, 64, 64)],
)
def test_matmul_4d(b: int, h: int, s: int, d: int) -> None:
    # Attention QK^T: [B, H, S, D] @ [B, H, D, S] → [B, H, S, S]
    q = torch.randn((b, h, s, d), dtype=torch.bfloat16)
    k = torch.randn((b, h, d, s), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.matmul, q, k, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "batch,m,k,n",
    [(2, 32, 64, 32), (4, 64, 32, 64)],
)
def test_bmm(batch: int, m: int, k: int, n: int) -> None:
    a = torch.randn((batch, m, k), dtype=torch.bfloat16)
    b = torch.randn((batch, k, n), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.bmm, a, b, atol=0.05, rtol=0.05)


def _matmul_grads(
    dev: str,
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
    g_cpu: torch.Tensor,
    need_self: bool = True,
    need_other: bool = True,
):
    """Grads of ``a @ b`` w.r.t. the inputs that require grad, evaluated on ``dev``.

    ``matmul``'s registered derivative is ``aten::matmul_backward``; on the tt
    device that dispatches to ``tt_matmul_backward``. Using ``autograd.grad``
    lets ``mask`` follow ``requires_grad`` and gives a CPU reference (CPU has no
    ``matmul_backward`` kernel, but its matmul derivative computes the same math).
    Inputs are materialized on CPU and copied to ``dev`` so both sides see
    identical data (the tt RNG doesn't share the CPU seed).
    """
    a = a_cpu.to(dev).detach().requires_grad_(need_self)
    b = b_cpu.to(dev).detach().requires_grad_(need_other)
    out = torch.matmul(a, b)
    inputs = tuple(t for t, need in ((a, need_self), (b, need_other)) if need)
    grads = torch.autograd.grad(out, inputs, g_cpu.to(dev))
    return [g.cpu() for g in grads]


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((32, 64), (64, 32)),  # 2D
        ((2, 32, 64), (2, 64, 32)),  # batched (no broadcast)
        ((2, 32, 64), (64, 32)),  # batched lhs, 2D rhs -> grad_other sums over batch
    ],
    ids=["2d", "batched", "broadcast_rhs"],
)
def test_matmul_backward(a_shape, b_shape) -> None:
    """aten::matmul_backward via autograd. The broadcast case exercises
    build_sum_to (grad_other reduced back to the un-broadcast rhs shape)."""
    a = torch.randn(a_shape, dtype=torch.bfloat16)
    b = torch.randn(b_shape, dtype=torch.bfloat16)
    g = torch.matmul(a, b)  # any tensor of the output shape works as grad_output
    g = torch.randn_like(g)

    cpu = _matmul_grads("cpu", a, b, g)
    tt = _matmul_grads("tt", a, b, g)
    for got, ref in zip(tt, cpu):
        torch.testing.assert_close(got, ref, atol=0.2, rtol=0.2)


@pytest.mark.parametrize(
    "need_self,need_other",
    [(True, False), (False, True)],
    ids=["self_only", "other_only"],
)
def test_matmul_backward_mask(need_self: bool, need_other: bool) -> None:
    """Only one grad requested: the mask passed to matmul_backward mirrors
    requires_grad, so the masked-off gradient is never built."""
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    b = torch.randn((64, 32), dtype=torch.bfloat16)
    g = torch.randn((32, 32), dtype=torch.bfloat16)

    cpu = _matmul_grads("cpu", a, b, g, need_self, need_other)
    tt = _matmul_grads("tt", a, b, g, need_self, need_other)
    assert len(tt) == 1
    torch.testing.assert_close(tt[0], cpu[0], atol=0.2, rtol=0.2)
