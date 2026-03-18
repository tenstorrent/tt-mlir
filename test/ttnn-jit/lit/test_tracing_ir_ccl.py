# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: ttnn-jit

"""
Lit tests for CCL (Collective Communication) ops tracing IR generation.

Covers all all_gather, all_reduce, and reduce_scatter use cases from example_ccls.mlir:
- ttir.all_reduce with cluster_axis=1, reduce_type=sum
- ttnn.reduce_scatter with cluster_axis=1, scatter_dim=2, reduce_type=sum
- ttnn.all_gather with all_gather_dim=2, cluster_axis=1
"""

import ttmlir.ir  # Ensure ttmlir is loaded before ttnn_jit to avoid TypeID conflicts
import ttnn
import torch
import inspect

from utils import create_sharded_tile_tensor

from ttnn_jit._src.ir_generator import generate_ir


def _get_tensor_args(func, *tensors):
    """Create tensor_args dict mapping param names to tensors."""
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    return {param_names[i]: t for i, t in enumerate(tensors)}


# ============================================================
# CCL operations - matching example_ccls.mlir
# ============================================================


def all_reduce_cluster_axis_1_func(a):
    """all_reduce with cluster_axis=1 (matches ttir.all_reduce in example_ccls.mlir)."""
    return ttnn.all_reduce(a, cluster_axis=1)


def all_gather_dim2_cluster_axis_1_func(a):
    """all_gather on dim=2, cluster_axis=1 (matches ttnn.all_gather in example_ccls.mlir)."""
    return ttnn.all_gather(a, dim=2, cluster_axis=1)


def reduce_scatter_dim2_cluster_axis_1_sum_func(a):
    """reduce_scatter dim=2, cluster_axis=1, reduce_type=sum (matches ttnn.reduce_scatter in example_ccls.mlir)."""
    return ttnn.reduce_scatter(a, dim=2, cluster_axis=1, reduce_type="sum")


# Additional CCL variants from example_ccls.mlir style (cluster_axis=0 variants for completeness)
def all_gather_dim0_cluster_axis_0_func(a):
    """all_gather dim=0, cluster_axis=0."""
    return ttnn.all_gather(a, dim=0, cluster_axis=0)


def all_reduce_cluster_axis_0_func(a):
    """all_reduce with cluster_axis=0."""
    return ttnn.all_reduce(a, cluster_axis=0)


def reduce_scatter_dim1_cluster_axis_0_func(a):
    """reduce_scatter dim=1, cluster_axis=0."""
    return ttnn.reduce_scatter(a, dim=1, cluster_axis=0)


def reduce_scatter_dim2_cluster_axis_1_max_func(a):
    """reduce_scatter dim=2, cluster_axis=1, reduce_type=max."""
    return ttnn.reduce_scatter(a, dim=2, cluster_axis=1, reduce_type="max")


def test_ir_generation(func, *tensors, debug=True):
    """Generate and print IR for a function."""
    tensor_args = _get_tensor_args(func, *tensors)
    ir, _ = generate_ir(func, debug, None, *tensors, _tensor_args=tensor_args)
    return ir


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    # 2D tensor for all_reduce (e.g. 1024x3584 -> 1024x3584 in example; we use 64x64 for test)
    input_2d = create_sharded_tile_tensor(device, (64, 64), (0, 0), torch.bfloat16)

    # 4D tensor for all_gather/reduce_scatter on dim=2 (e.g. 1x1x1024x3584 in example; we use 1x1x64x64)
    input_4d = create_sharded_tile_tensor(
        device, (1, 1, 64, 64), (0, 0), torch.bfloat16
    )

    # ----- example_ccls.mlir: ttir.all_reduce cluster_axis=1, reduce_type=sum -----
    # example_ccls.mlir: (1024x3584) -> (1024x3584); all_reduce preserves shape
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @all_reduce_cluster_axis_1_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]) -> tensor<64x64xbf16,
    # CHECK: return {{.*}} tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}
    test_ir_generation(all_reduce_cluster_axis_1_func, input_2d)

    # ----- example_ccls.mlir: ttnn.all_gather all_gather_dim=2, cluster_axis=1 -----
    # example_ccls.mlir: (1x1x256x3584) -> (1x1x1024x3584); output[dim] = input[dim]*mesh[cluster_axis]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @all_gather_dim2_cluster_axis_1_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xbf16, #ttnn_layout>]]) -> tensor<1x1x64x64xbf16,
    # CHECK: return {{.*}} tensor<1x1x64x64xbf16, #ttnn_layout{{[0-9]*}}
    test_ir_generation(all_gather_dim2_cluster_axis_1_func, input_4d)

    # ----- example_ccls.mlir: ttnn.reduce_scatter scatter_dim=2, cluster_axis=1, reduce_type=sum -----
    # example_ccls.mlir: (1x1x1024x3584) -> (1x1x256x3584); output[dim] = input[dim]/mesh[cluster_axis]
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @reduce_scatter_dim2_cluster_axis_1_sum_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xbf16, #ttnn_layout>]]) -> tensor<1x1x64x64xbf16,
    # CHECK: return {{.*}} tensor<1x1x64x64xbf16, #ttnn_layout{{[0-9]*}}
    test_ir_generation(reduce_scatter_dim2_cluster_axis_1_sum_func, input_4d)

    # ----- cluster_axis=0 variants (same style as existing test_tracing_ir CCL tests) -----
    # all_gather dim=0: output[0] = input[0]*mesh[0]; (64x64) -> (64x64) with mesh (1,1)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @all_gather_dim0_cluster_axis_0_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]) -> tensor<64x64xbf16,
    # CHECK: return {{.*}} tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}
    test_ir_generation(all_gather_dim0_cluster_axis_0_func, input_2d)

    # all_reduce: output shape = input shape; (64x64) -> (64x64)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @all_reduce_cluster_axis_0_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]) -> tensor<64x64xbf16,
    # CHECK: return {{.*}} tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}
    test_ir_generation(all_reduce_cluster_axis_0_func, input_2d)

    # reduce_scatter dim=1: output[1] = input[1]/mesh[0]; (64x64) -> (64x64) with mesh (1,1)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @reduce_scatter_dim1_cluster_axis_0_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+xbf16, #ttnn_layout>]]) -> tensor<64x64xbf16,
    # CHECK: return {{.*}} tensor<64x64xbf16, #ttnn_layout{{[0-9]*}}
    test_ir_generation(reduce_scatter_dim1_cluster_axis_0_func, input_2d)

    # reduce_scatter with reduce_type=max, dim=2, cluster_axis=1; (1x1x64x64) -> (1x1x64x64) with mesh (1,1)
    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @reduce_scatter_dim2_cluster_axis_1_max_func
    # CHECK-SAME: (%arg0: [[IN_TYPE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xbf16, #ttnn_layout>]]) -> tensor<1x1x64x64xbf16,
    # CHECK: return {{.*}} tensor<1x1x64x64xbf16, #ttnn_layout{{[0-9]*}}
    test_ir_generation(reduce_scatter_dim2_cluster_axis_1_max_func, input_4d)

    ttnn.close_device(device)
