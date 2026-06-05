"""Tests for aten::arange creation ops."""

import pytest
import torch

@pytest.mark.usefixtures("skip_if_sim")
def test_arange_end_only() -> None:
    expected = torch.arange(128, dtype=torch.float32)
    result = torch.arange(128, dtype=torch.float32, device="tt").cpu()
    torch.testing.assert_close(result, expected)


@pytest.mark.usefixtures("skip_if_sim")
def test_arange_start_end() -> None:
    expected = torch.arange(32, 96, dtype=torch.float32)
    result = torch.arange(32, 96, dtype=torch.float32, device="tt").cpu()
    torch.testing.assert_close(result, expected)


@pytest.mark.usefixtures("skip_if_sim")
def test_arange_start_end_step() -> None:
    expected = torch.arange(0, 64, 2, dtype=torch.float32)
    result = torch.arange(0, 64, 2, dtype=torch.float32, device="tt").cpu()
    torch.testing.assert_close(result, expected)


@pytest.mark.usefixtures("skip_if_sim")
def test_arange_long_dtype() -> None:
    # position_ids in Llama use long arange
    expected = torch.arange(128, dtype=torch.long)
    result = torch.arange(128, dtype=torch.long, device="tt").cpu()
    torch.testing.assert_close(result, expected)


def test_arange_bfloat16() -> None:
    expected = torch.arange(32, dtype=torch.bfloat16)
    result = torch.arange(32, dtype=torch.bfloat16, device="tt").cpu()
    torch.testing.assert_close(result, expected)
