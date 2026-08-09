"""Tests for aten::argmax."""

import pytest
import torch

from tt_kurbla.torch.testing import assert_close_cpu_vs_tt


@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_argmax_no_dim(shape: tuple) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    # Flatten reduction — result is a scalar-shaped tensor.
    assert_close_cpu_vs_tt(lambda x: torch.argmax(x), a)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_argmax_dim(dim: int) -> None:
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.argmax(x, dim=dim), a)


@pytest.mark.parametrize("dim", [0, 1])
def test_argmax_keepdim(dim: int) -> None:
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.argmax(x, dim=dim, keepdim=True), a)


def test_argmax_last_token() -> None:
    # Mirrors the decode sampling: logits[:, -1:, :].argmax(dim=-1)
    logits = torch.randn((1, 128, 256), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x[:, -1:, :].argmax(dim=-1), logits)


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_max_dim_values(dim: int, keepdim: bool) -> None:
    # max.dim returns (values, indices); check the values output. A per-row
    # permutation gives an unambiguous max so bf16 rounding can't flip the result.
    a = torch.stack([torch.randperm(64) for _ in range(32)]).to(torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.max(x, dim=dim, keepdim=keepdim).values, a)


@pytest.mark.parametrize("dim", [1, -1])
def test_max_dim_indices(dim: int) -> None:
    # max.dim indices == argmax over the same dim. Reduce within a row, where the
    # per-row permutation guarantees a unique max (no tie-break ambiguity).
    a = torch.stack([torch.randperm(64) for _ in range(32)]).to(torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.max(x, dim=dim).indices, a)
