# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: ttnn-jit

"""
Tests for the tracing frontend IR generation.

This test directly uses the TracingCompiler to generate TTIR and verify
the generated IR without going through the full JIT pass pipeline.
"""

import ttnn
import torch
import inspect

from utils import create_sharded_tile_tensor

# Import the IR generator directly
from ttnn_jit._src.ir_generator import generate_ir_from_tracing


def _get_tensor_args(func, *tensors):
    """Create tensor_args dict mapping param names to tensors."""
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    return {param_names[i]: t for i, t in enumerate(tensors)}


# ============================================================
# Test functions - these are the functions we'll compile
# ============================================================


def add_func(a, b):
    return ttnn.add(a, b)


def subtract_func(a, b):
    return ttnn.subtract(a, b)


def multiply_func(a, b):
    return ttnn.multiply(a, b)


def sum_func(a):
    return ttnn.sum(a, dim=0, keepdim=True)


def matmul_func(a, b):
    return ttnn.matmul(a, b)


def chained_ops_func(a, b):
    x = ttnn.add(a, b)
    y = ttnn.multiply(x, a)
    return ttnn.subtract(y, b)


def max_func(a):
    return ttnn.max(a, dim=1, keepdim=True)


def mean_func(a):
    return ttnn.mean(a, dim=0, keepdim=True)


# ============================================================
# Test runner
# ============================================================


def test_ir_generation(func, *tensors, debug=True):
    """Generate and print IR for a function."""
    tensor_args = _get_tensor_args(func, *tensors)
    # generate_ir_from_tracing(f, debug, *args, **kwargs)
    ir = generate_ir_from_tracing(func, debug, *tensors, _tensor_args=tensor_args)
    return ir


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    # Create test tensors
    input_a = create_sharded_tile_tensor(device, (64, 64), (0, 0), torch.bfloat16)
    input_b = create_sharded_tile_tensor(device, (64, 64), (0, 0), torch.bfloat16)
    input_c = create_sharded_tile_tensor(device, (64, 128), (0, 0), torch.bfloat16)

    # ============================================================
    # Binary operations tests
    # ============================================================

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @add_func
    # CHECK: ttir.add
    # CHECK: return
    test_ir_generation(add_func, input_a, input_b)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @subtract_func
    # CHECK: ttir.subtract
    # CHECK: return
    test_ir_generation(subtract_func, input_a, input_b)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @multiply_func
    # CHECK: ttir.multiply
    # CHECK: return
    test_ir_generation(multiply_func, input_a, input_b)

    # ============================================================
    # Reduction operations tests
    # ============================================================

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @sum_func
    # CHECK: ttir.sum
    # CHECK-SAME: dim_arg = [0 : i32]
    # CHECK-SAME: keep_dim = true
    # CHECK: return
    test_ir_generation(sum_func, input_a)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @max_func
    # CHECK: ttir.max
    # CHECK-SAME: dim_arg = [1 : i32]
    # CHECK-SAME: keep_dim = true
    # CHECK: return
    test_ir_generation(max_func, input_a)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @mean_func
    # CHECK: ttir.mean
    # CHECK-SAME: dim_arg = [0 : i32]
    # CHECK-SAME: keep_dim = true
    # CHECK: return
    test_ir_generation(mean_func, input_a)

    # ============================================================
    # Matmul test
    # ============================================================

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @matmul_func
    # CHECK: ttir.matmul
    # CHECK-SAME: transpose_a = false
    # CHECK-SAME: transpose_b = false
    # CHECK: return
    test_ir_generation(matmul_func, input_a, input_c)

    # ============================================================
    # Chained operations test
    # ============================================================

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @chained_ops_func
    # CHECK: ttir.add
    # CHECK: ttir.multiply
    # CHECK: ttir.subtract
    # CHECK: return
    test_ir_generation(chained_ops_func, input_a, input_b)

    ttnn.close_device(device)
