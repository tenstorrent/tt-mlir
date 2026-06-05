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
