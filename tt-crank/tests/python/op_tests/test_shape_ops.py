"""Tests for shape-manipulation ops: unsqueeze, squeeze, expand, transpose, permute, tril."""

import pytest
import torch

from tt_kurbla.torch.testing import assert_close_cpu_vs_tt


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_unsqueeze(dim: int) -> None:
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x.unsqueeze(dim), a)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_squeeze(dim: int) -> None:
    # Insert a size-1 dimension then squeeze it back out.
    a = torch.randn((32, 1, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x.squeeze(dim=1), a)


def test_squeeze_noop() -> None:
    # squeeze on a dim that is not size-1 is a no-op.
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x.squeeze(dim=0), a)


@pytest.mark.parametrize(
    "src_shape,target_shape",
    [
        ((1, 64), (32, 64)),
        ((32, 1), (32, 64)),
        ((1, 1, 64), (32, 32, 64)),
        ((1, 32, 64), (4, 32, 64)),
    ],
)
def test_expand(src_shape: tuple, target_shape: tuple) -> None:
    a = torch.randn(src_shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x.expand(target_shape), a)


@pytest.mark.parametrize(
    "shape,dim0,dim1",
    [
        ((32, 64), 0, 1),
        ((32, 64, 128), 0, 2),
        ((32, 64, 128), 1, 2),
        ((1, 8, 128, 64), 1, 2),
    ],
)
def test_transpose(shape: tuple, dim0: int, dim1: int) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x.transpose(dim0, dim1), a)


@pytest.mark.parametrize(
    "shape,perm",
    [
        ((32, 64, 128), (2, 0, 1)),
        ((32, 64, 128), (0, 2, 1)),
        ((1, 8, 128, 64), (0, 2, 1, 3)),
        ((2, 3, 4, 5), (3, 2, 1, 0)),
    ],
)
def test_permute(shape: tuple, perm: tuple) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: x.permute(perm), a)


@pytest.mark.parametrize("diagonal", [0, 1, -1])
@pytest.mark.parametrize("shape", [(32, 32), (64, 128), (32, 32, 64)])
def test_tril(shape: tuple, diagonal: int) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.tril(x, diagonal=diagonal), a)
