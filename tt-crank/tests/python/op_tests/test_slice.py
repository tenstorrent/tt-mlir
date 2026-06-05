"""Tests for aten::slice.Tensor (static slice along one dimension)."""

import pytest
import torch

from tt_kurbla.torch.testing import assert_close_cpu_vs_tt


@pytest.mark.parametrize(
    "shape,dim,start,end",
    [
        ((128, 64), 0, 0, 64),
        ((128, 64), 0, 32, 96),
        ((128, 64), 1, 0, 32),
        ((32, 128, 64), 1, 0, 64),
        ((32, 128, 64), 2, 32, 64),
        ((1, 8, 128, 64), 2, 0, 64),   # first half of seq dim
        ((1, 8, 128, 64), 2, 64, 128), # second half of seq dim
    ],
)
def test_slice(shape: tuple, dim: int, start: int, end: int) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x[..., start:end] if dim == len(shape) - 1
                           else x.narrow(dim, start, end - start), a)


def test_slice_ellipsis_last_dim() -> None:
    # RoPE rotate_half splits the last dim: x[..., :half] and x[..., half:]
    a = torch.randn((1, 8, 128, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x[..., :32], a)
    assert_close_cpu_vs_tt(lambda x: x[..., 32:], a)


def test_slice_negative_index() -> None:
    a = torch.randn((1, 8, 128, 64), dtype=torch.bfloat16)
    # Select last token: [:, :, -1:, :]
    assert_close_cpu_vs_tt(lambda x: x[:, :, -1:, :], a)


def test_slice_full_dim_is_noop() -> None:
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x[:, :], a)
