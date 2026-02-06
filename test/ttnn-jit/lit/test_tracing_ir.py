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
from ttnn_jit._src.ir_generator import generate_ir

# Import ops from shared definitions (aliased to match FileCheck patterns)
from op_definitions import (
    exp as exp_func,
    neg as neg_func,
    relu as relu_func,
    sqrt as sqrt_func,
    add as add_func,
    sub as subtract_func,
    mul as multiply_func,
    div as divide_func,
    matmul as matmul_func,
)


def _get_tensor_args(func, *tensors):
    """Create tensor_args dict mapping param names to tensors."""
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    return {param_names[i]: t for i, t in enumerate(tensors)}


# ============================================================
# Reduction operations - kept local due to specific parameters
# ============================================================


def sum_func(a):
    return ttnn.sum(a, dim=0, keepdim=True)


def max_func(a):
    return ttnn.max(a, dim=1, keepdim=True)


def mean_func(a):
    return ttnn.mean(a, dim=0, keepdim=False)


def sum_all_func(a):
    """Full reduction - sum over all dimensions without keepdim (scalar result)."""
    return ttnn.sum(a)


def sum_all_keepdim_func(a):
    """Full reduction - sum over all dimensions with keepdim=True."""
    return ttnn.sum(a, keepdim=True)


# ============================================================
# Composite/chained operations - kept local due to specific patterns
# ============================================================


def chained_ops_func(a, b):
    x = ttnn.add(a, b)
    y = ttnn.multiply(x, a)
    return ttnn.subtract(b, y)


# ============================================================
# Control flow test functions
# ============================================================


def loop_func(a, b):
    """Test function with a for loop - applies add 3 times."""
    result = a
    for _ in range(3):
        result = ttnn.add(result, b)
    return result


def conditional_true_func(a, b):
    """Test function with conditional - use_exp=True path."""
    use_exp = True
    if use_exp:
        return ttnn.exp(a)
    else:
        return ttnn.neg(b)


def conditional_false_func(a, b):
    """Test function with conditional - use_exp=False path."""
    use_exp = False
    if use_exp:
        return ttnn.exp(a)
    else:
        return ttnn.neg(b)


def loop_with_conditional_func(a, b):
    """Test function with loop and conditional combined."""
    result = a
    for i in range(2):
        if i % 2 == 0:
            result = ttnn.add(result, b)
        else:
            result = ttnn.multiply(result, b)
    return result


# ============================================================
# Test runner
# ============================================================


def test_ir_generation(func, *tensors, debug=True):
    """Generate and print IR for a function."""
    tensor_args = _get_tensor_args(func, *tensors)
    ir = generate_ir(func, debug, None, *tensors, _tensor_args=tensor_args)
    return ir


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    # Create test tensors
    input_a = create_sharded_tile_tensor(device, (64, 64), (0, 0), torch.bfloat16)
    input_b = create_sharded_tile_tensor(device, (64, 64), (0, 0), torch.bfloat16)
    input_c = create_sharded_tile_tensor(device, (64, 128), (0, 0), torch.bfloat16)
    input_d = create_sharded_tile_tensor(device, (128, 32), (0, 0), torch.bfloat16)

    # ============================================================
    # Unary operations tests
    # ============================================================

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @exp
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[VAL:[0-9]+]] = "ttir.exp"(%arg0)
    # CHECK-SAME: ([[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[VAL]] : [[IN_TYPE]]
    test_ir_generation(exp_func, input_a)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @neg
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[VAL:[0-9]+]] = "ttir.neg"(%arg0)
    # CHECK-SAME: ([[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[VAL]] : [[IN_TYPE]]
    test_ir_generation(neg_func, input_a)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @relu
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[VAL:[0-9]+]] = "ttir.relu"(%arg0)
    # CHECK-SAME: ([[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[VAL]] : [[IN_TYPE]]
    test_ir_generation(relu_func, input_a)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @sqrt
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[VAL:[0-9]+]] = "ttir.sqrt"(%arg0)
    # CHECK-SAME: ([[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[VAL]] : [[IN_TYPE]]
    test_ir_generation(sqrt_func, input_a)

    # ============================================================
    # Binary operations tests
    # ============================================================

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @add
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]
    # CHECK-SAME: %arg1: [[IN_TYPE]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[VAL:[0-9]+]] = "ttir.add"(%arg0, %arg1)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[VAL]] : [[IN_TYPE]]
    test_ir_generation(add_func, input_a, input_b)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @sub
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]
    # CHECK-SAME: %arg1: [[IN_TYPE]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[VAL:[0-9]+]] = "ttir.subtract"(%arg0, %arg1)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[VAL]] : [[IN_TYPE]]
    test_ir_generation(subtract_func, input_a, input_b)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @mul
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]
    # CHECK-SAME: %arg1: [[IN_TYPE]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[VAL:[0-9]+]] = "ttir.multiply"(%arg0, %arg1)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[VAL]] : [[IN_TYPE]]
    test_ir_generation(multiply_func, input_a, input_b)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @div
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]
    # CHECK-SAME: %arg1: [[IN_TYPE]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[VAL:[0-9]+]] = "ttir.div"(%arg0, %arg1)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[VAL]] : [[IN_TYPE]]
    test_ir_generation(divide_func, input_a, input_b)

    # ============================================================
    # Reduction operations tests
    # ============================================================

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @sum_func
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<1x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.sum"(%arg0)
    # CHECK-SAME: dim_arg = [0 : i32]
    # CHECK-SAME: keep_dim = true
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>) -> tensor<1x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<1x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(sum_func, input_a)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @max_func
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<64x1xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.max"(%arg0)
    # CHECK-SAME: dim_arg = [1 : i32]
    # CHECK-SAME: keep_dim = true
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x1xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<64x1xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(max_func, input_a)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @mean_func
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.mean"(%arg0)
    # CHECK-SAME: dim_arg = [0 : i32]
    # CHECK-SAME: keep_dim = false
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>) -> tensor<64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(mean_func, input_a)

    # Full reduction without keepdim - scalar result (0D tensor)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @sum_all_func
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<bf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.sum"(%arg0)
    # CHECK-SAME: keep_dim = false
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>) -> tensor<bf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<bf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(sum_all_func, input_a)

    # Full reduction with keepdim=True - all dims become 1
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @sum_all_keepdim_func
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<1x1xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.sum"(%arg0)
    # CHECK-SAME: keep_dim = true
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>) -> tensor<1x1xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<1x1xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(sum_all_keepdim_func, input_a)

    # ============================================================
    # Matmul test
    # ============================================================

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @matmul
    # CHECK-SAME: (%arg0: tensor<64x128xbf16, #ttnn_layout>
    # CHECK-SAME: %arg1: tensor<128x32xbf16, #ttnn_layout1>)
    # CHECK-SAME: -> tensor<64x32xbf16, #ttnn_layout2>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.matmul"(%arg0, %arg1)
    # CHECK-SAME: transpose_a = false
    # CHECK-SAME: transpose_b = false
    # CHECK-SAME: (tensor<64x128xbf16, #ttnn_layout>, tensor<128x32xbf16, #ttnn_layout1>) -> tensor<64x32xbf16, #ttnn_layout2>
    # CHECK: return %[[VAL]] : tensor<64x32xbf16, #ttnn_layout2>
    test_ir_generation(matmul_func, input_c, input_d)

    # ============================================================
    # Chained operations test
    # ============================================================

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @chained_ops_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]
    # CHECK-SAME: %arg1: [[IN_TYPE]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[V0:[0-9]+]] = "ttir.add"(%arg0, %arg1)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: %[[V1:[0-9]+]] = "ttir.multiply"(%[[V0]], %arg0)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: %[[V2:[0-9]+]] = "ttir.subtract"(%arg1, %[[V1]])
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[V2]] : [[IN_TYPE]]
    test_ir_generation(chained_ops_func, input_a, input_b)

    # ============================================================
    # Loop test - for loop unrolled to 3 adds
    # ============================================================

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @loop_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]
    # CHECK-SAME: %arg1: [[IN_TYPE]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[V0:[0-9]+]] = "ttir.add"(%arg0, %arg1)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: %[[V1:[0-9]+]] = "ttir.add"(%[[V0]], %arg1)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: %[[V2:[0-9]+]] = "ttir.add"(%[[V1]], %arg1)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[V2]] : [[IN_TYPE]]
    test_ir_generation(loop_func, input_a, input_b)

    # ============================================================
    # Conditional tests - if statement with boolean flag
    # ============================================================

    # Test with use_exp=True - should generate ttir.exp
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @conditional_true_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]
    # CHECK-SAME: %arg1: [[IN_TYPE]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[VAL:[0-9]+]] = "ttir.exp"(%arg0)
    # CHECK-SAME: ([[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[VAL]] : [[IN_TYPE]]
    test_ir_generation(conditional_true_func, input_a, input_b)

    # Test with use_exp=False - should generate ttir.neg
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @conditional_false_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]
    # CHECK-SAME: %arg1: [[IN_TYPE]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[VAL:[0-9]+]] = "ttir.neg"(%arg1)
    # CHECK-SAME: ([[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[VAL]] : [[IN_TYPE]]
    test_ir_generation(conditional_false_func, input_a, input_b)

    # ============================================================
    # Loop with conditional - combines both control flow patterns
    # ============================================================

    # Loop runs twice: i=0 (add), i=1 (multiply)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @loop_with_conditional_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]
    # CHECK-SAME: %arg1: [[IN_TYPE]])
    # CHECK-SAME: -> [[IN_TYPE]]
    # CHECK: %[[V0:[0-9]+]] = "ttir.add"(%arg0, %arg1)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: %[[V1:[0-9]+]] = "ttir.multiply"(%[[V0]], %arg1)
    # CHECK-SAME: ([[IN_TYPE]], [[IN_TYPE]]) -> [[IN_TYPE]]
    # CHECK: return %[[V1]] : [[IN_TYPE]]
    test_ir_generation(loop_with_conditional_func, input_a, input_b)

    ttnn.close_device(device)
