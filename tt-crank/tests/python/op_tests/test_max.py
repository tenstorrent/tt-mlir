# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for aten::amax and the values of aten::max.dim, eager and compiled."""

import pytest
import torch

from tt_crank.torch.testing import ExecutionMode, assert_close_cpu_vs_tt

_MODES = [ExecutionMode.EAGER, ExecutionMode.COMPILE]


def _rows() -> torch.Tensor:
    # Per-row permutations give a unique max, so bf16 rounding cannot flip the result.
    return torch.stack([torch.randperm(64) for _ in range(32)]).to(torch.bfloat16)


@pytest.mark.parametrize("mode", _MODES, ids=lambda m: m.name.lower())
@pytest.mark.parametrize("dim", [0, 1, -1, (0, 1)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_amax(dim, keepdim: bool, mode: ExecutionMode) -> None:
    assert_close_cpu_vs_tt(
        lambda x: torch.amax(x, dim=dim, keepdim=keepdim), _rows(), mode=mode
    )


@pytest.mark.parametrize("mode", _MODES, ids=lambda m: m.name.lower())
def test_amax_all_dims(mode: ExecutionMode) -> None:
    a = torch.randn((4, 32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.amax(x), a, mode=mode)


@pytest.mark.parametrize("mode", _MODES, ids=lambda m: m.name.lower())
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_max_dim_values(dim: int, keepdim: bool, mode: ExecutionMode) -> None:
    assert_close_cpu_vs_tt(
        lambda x: torch.max(x, dim=dim, keepdim=keepdim).values, _rows(), mode=mode
    )


def test_max_dim_values_f32() -> None:
    a = torch.randn((32, 64), dtype=torch.float32)
    assert_close_cpu_vs_tt(lambda x: torch.max(x, dim=1).values, a)
