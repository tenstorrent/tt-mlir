# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for aten::cat (concatenation) including the empty-tensor edge case."""

import pytest
import torch

from tt_crank.torch.testing import assert_close_cpu_vs_tt


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cat_two_tensors(dim: int) -> None:
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    b = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x, y: torch.cat([x, y], dim=dim), a, b)


def test_cat_three_tensors() -> None:
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    b = torch.randn((32, 64), dtype=torch.bfloat16)
    c = torch.randn((32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x, y, z: torch.cat([x, y, z], dim=0), a, b, c)


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_cat_3d(dim: int) -> None:
    a = torch.randn((2, 32, 64), dtype=torch.bfloat16)
    b = torch.randn((2, 32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x, y: torch.cat([x, y], dim=dim), a, b)


def test_cat_seq_dim_4d() -> None:
    # Mirrors the DynamicCache KV-cache concat: [B, H, S1, D] + [B, H, S2, D] -> [B, H, S1+S2, D]
    a = torch.randn((1, 8, 128, 64), dtype=torch.bfloat16)
    b = torch.randn((1, 8, 32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x, y: torch.cat([x, y], dim=2), a, b)


def test_cat_empty_tensor_ignored() -> None:
    # CPU cat drops zero-numel tensors regardless of rank; our kernel must match.
    empty = torch.tensor([], dtype=torch.bfloat16)  # rank 1, shape [0]
    a = torch.randn((32, 64), dtype=torch.bfloat16)
    empty_tt = empty.to("tt")
    a_tt = a.to("tt")
    result = torch.cat([empty_tt, a_tt], dim=-2).cpu()
    torch.testing.assert_close(result, a)
