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
    concat_dim0 as concat_dim0_func,
    concat_dim1 as concat_dim1_func,
    concat_three as concat_three_func,
    repeat_2x1 as repeat_2x1_func,
    repeat_1x3 as repeat_1x3_func,
    repeat_2x2 as repeat_2x2_func,
    embedding as embedding_func,
    gather_dim0 as gather_dim0_func,
    gather_dim1 as gather_dim1_func,
)


def _get_tensor_args(func, *tensors):
    """Create tensor_args dict mapping param names to tensors."""
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    return {param_names[i]: t for i, t in enumerate(tensors)}


# ============================================================
# Tensor manipulation (TM) operations - permute tests
# ============================================================


def permute_2d_func(a):
    """2D transpose - swap dimensions."""
    return ttnn.permute(a, permutation=[1, 0])


def permute_4d_func(a):
    """4D attention-style transpose - BHSD -> BSHD."""
    return ttnn.permute(a, permutation=[0, 2, 1, 3])


# ============================================================
# Tensor manipulation (TM) operations - transpose tests
# ============================================================


def transpose_2d_func(a):
    """2D transpose - swap dimensions 0 and 1."""
    return ttnn.transpose(a, dim0=0, dim1=1)


def transpose_4d_func(a):
    """4D transpose - swap dimensions 1 and 2 (e.g., BHSD -> BSHD)."""
    return ttnn.transpose(a, dim0=1, dim1=2)


def transpose_negative_dims_func(a):
    """Transpose with negative dimensions - swap last two dims."""
    return ttnn.transpose(a, dim0=-2, dim1=-1)


# ============================================================
# Tensor manipulation (TM) operations - reshape tests
# ============================================================


def reshape_2d_to_1d_func(a):
    """Flatten 2D tensor to 1D."""
    return ttnn.reshape(a, shape=[64 * 64])


def reshape_2d_to_3d_func(a):
    """Reshape 2D tensor to 3D."""
    return ttnn.reshape(a, shape=[1, 64, 64])


def reshape_4d_to_2d_func(a):
    """Reshape 4D tensor to 2D (flatten batch and spatial)."""
    return ttnn.reshape(a, shape=[1 * 8 * 32, 64])


# ============================================================
# Tensor manipulation (TM) operations - rearrange tests
# ============================================================


def rearrange_identity_func(a):
    """Identity rearrange - no change in shape."""
    return ttnn.rearrange(a, pattern="b h w c -> b h w c")


def rearrange_reorder_func(a):
    """Reordering dimensions - BHWC -> BCHW."""
    return ttnn.rearrange(a, pattern="b h w c -> b c h w")


def rearrange_merge_func(a):
    """Merge two dimensions into one using rearrange."""
    return ttnn.rearrange(a, pattern="b h w c -> (b h) w c")


def rearrange_reorder_merge_func(a):
    """Reorder and merge dimensions - 'b h w c -> h (b w) c'."""
    return ttnn.rearrange(a, pattern="b h w c -> h (b w) c")


def rearrange_multi_merge_func(a):
    """Merge multiple dimensions - 'b h w c -> b (c h w)'."""
    return ttnn.rearrange(a, pattern="b h w c -> b (c h w)")


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

    # ============================================================
    # Tensor manipulation (TM) operations tests - permute
    # ============================================================

    # 2D permute (transpose)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @permute_2d_func
    # CHECK-SAME: (%arg0: tensor<64x128xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.permute"(%arg0)
    # CHECK-SAME: permutation = array<i64: 1, 0>
    # CHECK-SAME: (tensor<64x128xbf16, #ttnn_layout>) -> tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(permute_2d_func, input_c)

    # 4D permute (attention-style BHSD -> BSHD)
    input_4d = create_sharded_tile_tensor(
        device, (1, 8, 32, 64), (0, 0), torch.bfloat16
    )
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @permute_4d_func
    # CHECK-SAME: (%arg0: tensor<1x8x32x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<1x32x8x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.permute"(%arg0)
    # CHECK-SAME: permutation = array<i64: 0, 2, 1, 3>
    # CHECK-SAME: (tensor<1x8x32x64xbf16, #ttnn_layout>) -> tensor<1x32x8x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<1x32x8x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(permute_4d_func, input_4d)

    # ============================================================
    # Tensor manipulation (TM) operations tests - transpose
    # ============================================================

    # 2D transpose (swap dims 0 and 1)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @transpose_2d_func
    # CHECK-SAME: (%arg0: tensor<64x128xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.transpose"(%arg0)
    # CHECK-SAME: dim0 = 0 : si32
    # CHECK-SAME: dim1 = 1 : si32
    # CHECK-SAME: (tensor<64x128xbf16, #ttnn_layout>) -> tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(transpose_2d_func, input_c)

    # 4D transpose (swap dims 1 and 2 - BHSD -> BSHD style)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @transpose_4d_func
    # CHECK-SAME: (%arg0: tensor<1x8x32x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<1x32x8x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.transpose"(%arg0)
    # CHECK-SAME: dim0 = 1 : si32
    # CHECK-SAME: dim1 = 2 : si32
    # CHECK-SAME: (tensor<1x8x32x64xbf16, #ttnn_layout>) -> tensor<1x32x8x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<1x32x8x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(transpose_4d_func, input_4d)

    # Transpose with negative dimensions (swap last two dims of 4D tensor)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @transpose_negative_dims_func
    # CHECK-SAME: (%arg0: tensor<1x8x32x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<1x8x64x32xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.transpose"(%arg0)
    # CHECK-SAME: dim0 = -2 : si32
    # CHECK-SAME: dim1 = -1 : si32
    # CHECK-SAME: (tensor<1x8x32x64xbf16, #ttnn_layout>) -> tensor<1x8x64x32xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<1x8x64x32xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(transpose_negative_dims_func, input_4d)

    # ============================================================
    # Tensor manipulation (TM) operations tests - reshape
    # ============================================================

    # Reshape 2D to 1D (flatten)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @reshape_2d_to_1d_func
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<4096xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.reshape"(%arg0)
    # CHECK-SAME: shape = [4096 : i32]
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>) -> tensor<4096xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<4096xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(reshape_2d_to_1d_func, input_a)

    # Reshape 2D to 3D (add batch dimension)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @reshape_2d_to_3d_func
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<1x64x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.reshape"(%arg0)
    # CHECK-SAME: shape = [1 : i32, 64 : i32, 64 : i32]
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>) -> tensor<1x64x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<1x64x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(reshape_2d_to_3d_func, input_a)

    # Reshape 4D to 2D (flatten batch and spatial)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @reshape_4d_to_2d_func
    # CHECK-SAME: (%arg0: tensor<1x8x32x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<256x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.reshape"(%arg0)
    # CHECK-SAME: shape = [256 : i32, 64 : i32]
    # CHECK-SAME: (tensor<1x8x32x64xbf16, #ttnn_layout>) -> tensor<256x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<256x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(reshape_4d_to_2d_func, input_4d)

    # ============================================================
    # Tensor manipulation (TM) operations tests - rearrange
    # All 5 supported rearrange patterns are tested here:
    # 1. Identity: 'b h w c -> b h w c'
    # 2. Reordering: 'b h w c -> b c h w'
    # 3. Merging: 'b h w c -> (b h) w c'
    # 4. Reorder + merge: 'b h w c -> h (b w) c'
    # 5. Multi-merge: 'b h w c -> b (c h w)'
    # ============================================================

    # Create 4D BHWC tensor for rearrange tests: (b=2, h=4, w=32, c=64)
    input_bhwc = create_sharded_tile_tensor(
        device, (2, 4, 32, 64), (0, 0), torch.bfloat16
    )

    # 1. Identity rearrange: 'b h w c -> b h w c' (no shape change)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @rearrange_identity_func
    # CHECK-SAME: (%arg0: tensor<2x4x32x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<2x4x32x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.rearrange"(%arg0)
    # CHECK-SAME: pattern = "b h w c -> b h w c"
    # CHECK-SAME: (tensor<2x4x32x64xbf16, #ttnn_layout>) -> tensor<2x4x32x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<2x4x32x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(rearrange_identity_func, input_bhwc)

    # 2. Reordering: 'b h w c -> b c h w' (BHWC -> BCHW)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @rearrange_reorder_func
    # CHECK-SAME: (%arg0: tensor<2x4x32x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<2x64x4x32xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.rearrange"(%arg0)
    # CHECK-SAME: pattern = "b h w c -> b c h w"
    # CHECK-SAME: (tensor<2x4x32x64xbf16, #ttnn_layout>) -> tensor<2x64x4x32xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<2x64x4x32xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(rearrange_reorder_func, input_bhwc)

    # 3. Merging: 'b h w c -> (b h) w c' (merge batch and height)
    # Output shape: (b*h, w, c) = (2*4, 32, 64) = (8, 32, 64)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @rearrange_merge_func
    # CHECK-SAME: (%arg0: tensor<2x4x32x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<8x32x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.rearrange"(%arg0)
    # CHECK-SAME: pattern = "b h w c -> (b h) w c"
    # CHECK-SAME: (tensor<2x4x32x64xbf16, #ttnn_layout>) -> tensor<8x32x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<8x32x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(rearrange_merge_func, input_bhwc)

    # 4. Reorder + merge: 'b h w c -> h (b w) c'
    # Output shape: (h, b*w, c) = (4, 2*32, 64) = (4, 64, 64)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @rearrange_reorder_merge_func
    # CHECK-SAME: (%arg0: tensor<2x4x32x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<4x64x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.rearrange"(%arg0)
    # CHECK-SAME: pattern = "b h w c -> h (b w) c"
    # CHECK-SAME: (tensor<2x4x32x64xbf16, #ttnn_layout>) -> tensor<4x64x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<4x64x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(rearrange_reorder_merge_func, input_bhwc)

    # 5. Multi-merge: 'b h w c -> b (c h w)' (flatten to 2D)
    # Output shape: (b, c*h*w) = (2, 64*4*32) = (2, 8192)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @rearrange_multi_merge_func
    # CHECK-SAME: (%arg0: tensor<2x4x32x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<2x8192xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.rearrange"(%arg0)
    # CHECK-SAME: pattern = "b h w c -> b (c h w)"
    # CHECK-SAME: (tensor<2x4x32x64xbf16, #ttnn_layout>) -> tensor<2x8192xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<2x8192xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(rearrange_multi_merge_func, input_bhwc)

    # ============================================================
    # Concat operations tests
    # ============================================================

    # Concat along dimension 0: [64, 64] + [64, 64] -> [128, 64]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @concat_dim0
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>
    # CHECK-SAME: %arg1: tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}>)
    # CHECK-SAME: -> tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.concat"(%arg0, %arg1)
    # CHECK-SAME: dim = 0 : si32
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>, tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}>) -> tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(concat_dim0_func, input_a, input_b)

    # Concat along dimension 1: [64, 64] + [64, 64] -> [64, 128]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @concat_dim1
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>
    # CHECK-SAME: %arg1: tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}>)
    # CHECK-SAME: -> tensor<64x128xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.concat"(%arg0, %arg1)
    # CHECK-SAME: dim = 1 : si32
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>, tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}>) -> tensor<64x128xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<64x128xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(concat_dim1_func, input_a, input_b)

    # Concat three tensors along dimension 0: [64, 64] + [64, 64] + [64, 64] -> [192, 64]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @concat_three
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>
    # CHECK-SAME: %arg1: tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK-SAME: %arg2: tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}>)
    # CHECK-SAME: -> tensor<192x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.concat"(%arg0, %arg1, %arg2)
    # CHECK-SAME: dim = 0 : si32
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>, tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}>, tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}>) -> tensor<192x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<192x64xbf16, #ttnn_layout{{[0-9]*}}>
    input_c_same = create_sharded_tile_tensor(device, (64, 64), (0, 0), torch.bfloat16)
    test_ir_generation(concat_three_func, input_a, input_b, input_c_same)

    # ============================================================
    # Repeat operations tests
    # ============================================================

    # Repeat [2, 1]: [64, 64] -> [128, 64]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @repeat_2x1
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.repeat"(%arg0)
    # CHECK-SAME: repeat_dimensions = array<i64: 2, 1>
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>) -> tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<128x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(repeat_2x1_func, input_a)

    # Repeat [1, 3]: [64, 64] -> [64, 192]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @repeat_1x3
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<64x192xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.repeat"(%arg0)
    # CHECK-SAME: repeat_dimensions = array<i64: 1, 3>
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x192xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<64x192xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(repeat_1x3_func, input_a)

    # Repeat [2, 2]: [64, 64] -> [128, 128]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @repeat_2x2
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>)
    # CHECK-SAME: -> tensor<128x128xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.repeat"(%arg0)
    # CHECK-SAME: repeat_dimensions = array<i64: 2, 2>
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>) -> tensor<128x128xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<128x128xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(repeat_2x2_func, input_a)

    # ============================================================
    # Embedding operations tests
    # ============================================================

    # Create tensors for embedding test
    # indices: [32, 32] (batch of indices)
    # weight: [512, 128] (embedding table with 512 entries of dim 128)
    # output: [32, 32, 128]
    input_indices = create_sharded_tile_tensor(device, (32, 32), (0, 0), torch.bfloat16)
    weight_table = create_sharded_tile_tensor(
        device, (512, 128), (0, 0), torch.bfloat16
    )

    # Embedding: [32, 32] indices + [512, 128] weight -> [32, 32, 128]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @embedding
    # CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout>
    # CHECK-SAME: %arg1: tensor<512x128xbf16, #ttnn_layout{{[0-9]*}}>)
    # CHECK-SAME: -> tensor<32x32x128xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.embedding"(%arg0, %arg1)
    # CHECK-SAME: (tensor<32x32xbf16, #ttnn_layout>, tensor<512x128xbf16, #ttnn_layout{{[0-9]*}}>) -> tensor<32x32x128xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<32x32x128xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(embedding_func, input_indices, weight_table)

    # ============================================================
    # Gather operations tests
    # ============================================================

    # Create tensors for gather test
    # For torch.gather, output shape = index shape
    # input: [64, 64], index: [64, 32], dim=1 -> output: [64, 32]
    gather_input = create_sharded_tile_tensor(device, (64, 64), (0, 0), torch.bfloat16)
    gather_index_dim0 = create_sharded_tile_tensor(
        device, (32, 64), (0, 0), torch.bfloat16
    )
    gather_index_dim1 = create_sharded_tile_tensor(
        device, (64, 32), (0, 0), torch.bfloat16
    )

    # Gather along dimension 0: [64, 64] input + [32, 64] index -> [32, 64]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @gather_dim0
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>
    # CHECK-SAME: %arg1: tensor<32x64xbf16, #ttnn_layout{{[0-9]*}}>)
    # CHECK-SAME: -> tensor<32x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.gather"(%arg0, %arg1)
    # CHECK-SAME: collapsed_slice_dims = array<i64: 0, 1>
    # CHECK-SAME: index_vector_dim = 2 : si64
    # CHECK-SAME: offset_dims = array<i64>
    # CHECK-SAME: slice_sizes = array<i64: 1, 1>
    # CHECK-SAME: start_index_map = array<i64: 0>
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>, tensor<32x64xbf16, #ttnn_layout{{[0-9]*}}>) -> tensor<32x64xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<32x64xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(gather_dim0_func, gather_input, gather_index_dim0)

    # Gather along dimension 1: [64, 64] input + [64, 32] index -> [64, 32]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @gather_dim1
    # CHECK-SAME: (%arg0: tensor<64x64xbf16, #ttnn_layout>
    # CHECK-SAME: %arg1: tensor<64x32xbf16, #ttnn_layout{{[0-9]*}}>)
    # CHECK-SAME: -> tensor<64x32xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: %[[VAL:[0-9]+]] = "ttir.gather"(%arg0, %arg1)
    # CHECK-SAME: collapsed_slice_dims = array<i64: 0, 1>
    # CHECK-SAME: index_vector_dim = 2 : si64
    # CHECK-SAME: offset_dims = array<i64>
    # CHECK-SAME: slice_sizes = array<i64: 1, 1>
    # CHECK-SAME: start_index_map = array<i64: 1>
    # CHECK-SAME: (tensor<64x64xbf16, #ttnn_layout>, tensor<64x32xbf16, #ttnn_layout{{[0-9]*}}>) -> tensor<64x32xbf16, #ttnn_layout{{[0-9]*}}>
    # CHECK: return %[[VAL]] : tensor<64x32xbf16, #ttnn_layout{{[0-9]*}}>
    test_ir_generation(gather_dim1_func, gather_input, gather_index_dim1)

    ttnn.close_device(device)
