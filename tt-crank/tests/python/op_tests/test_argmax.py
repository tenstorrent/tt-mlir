"""Tests for aten::argmax."""

import pytest
import torch

from tt_kurbla.torch.testing import assert_close_cpu_vs_tt


@pytest.mark.usefixtures("skip_if_sim")
@pytest.mark.parametrize("shape", [(32, 64), (32, 64, 128)])
def test_argmax_no_dim(shape: tuple) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    # Flatten reduction — result is a scalar-shaped tensor.
    assert_close_cpu_vs_tt(lambda x: torch.argmax(x), a)


@pytest.mark.usefixtures("skip_if_sim")
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_argmax_dim(dim: int) -> None:
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.argmax(x, dim=dim), a)


@pytest.mark.usefixtures("skip_if_sim")
@pytest.mark.parametrize("dim", [0, 1])
def test_argmax_keepdim(dim: int) -> None:
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.argmax(x, dim=dim, keepdim=True), a)


@pytest.mark.usefixtures("skip_if_sim")
def test_argmax_last_token() -> None:
    # Mirrors the decode sampling: logits[:, -1:, :].argmax(dim=-1)
    logits = torch.randn((1, 128, 256), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x[:, -1:, :].argmax(dim=-1), logits)
