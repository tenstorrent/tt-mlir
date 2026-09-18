# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for aten::arange creation ops."""

import pytest
import torch


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.long, None], ids=["float32", "long", "default"]
)
def test_arange_end_only(dtype: torch.dtype | None) -> None:
    # dtype=None exercises the default: integer bounds must yield int64.
    expected = torch.arange(128, dtype=dtype)
    result = torch.arange(128, dtype=dtype, device="tt").cpu()
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.long, None], ids=["float32", "long", "default"]
)
def test_arange_start_end(dtype: torch.dtype | None) -> None:
    expected = torch.arange(32, 96, dtype=dtype)
    result = torch.arange(32, 96, dtype=dtype, device="tt").cpu()
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.long, None], ids=["float32", "long", "default"]
)
def test_arange_start_end_step(dtype: torch.dtype | None) -> None:
    expected = torch.arange(0, 64, 2, dtype=dtype)
    result = torch.arange(0, 64, 2, dtype=dtype, device="tt").cpu()
    torch.testing.assert_close(result, expected)


def test_arange_bfloat16() -> None:
    expected = torch.arange(32, dtype=torch.bfloat16)
    result = torch.arange(32, dtype=torch.bfloat16, device="tt").cpu()
    torch.testing.assert_close(result, expected)
