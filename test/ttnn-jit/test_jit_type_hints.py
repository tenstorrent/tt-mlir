# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for JIT-compiled functions that use ttnn type hints (e.g. ttnn.Tensor)."""

import ttnn_jit
import ttnn
import torch
import pytest

from utils import create_dram_tensor


@ttnn_jit.jit(compile_only=True)
def single_exp_typed(input: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.exp(input)


@ttnn_jit.jit(compile_only=True)
def add_typed(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.add(a, b)


@ttnn_jit.jit(compile_only=True)
def single_exp_untyped(input):
    return ttnn.exp(input)


def test_jit_compile_only_with_type_hints(device):
    """JIT with type hints (ttnn.Tensor) should compile and return IR, not raise AttributeError."""
    input_tensor = create_dram_tensor(device, (64, 128), torch.bfloat16)
    # compile_only=True returns the IR module only
    module = single_exp_typed(input_tensor)
    assert module is not None


def test_jit_compile_only_untyped_still_works(device):
    """JIT without type hints should continue to work."""
    input_tensor = create_dram_tensor(device, (64, 128), torch.bfloat16)
    module = single_exp_untyped(input_tensor)
    assert module is not None


def test_jit_compile_only_add_with_type_hints(device):
    """JIT add with type hints should compile."""
    a = create_dram_tensor(device, (64, 128), torch.bfloat16)
    b = create_dram_tensor(device, (64, 128), torch.bfloat16)
    module = add_typed(a, b)
    assert module is not None
