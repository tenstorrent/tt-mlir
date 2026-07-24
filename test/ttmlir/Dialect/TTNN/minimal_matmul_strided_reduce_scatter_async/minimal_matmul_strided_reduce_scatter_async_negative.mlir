// RUN: not ttmlir-opt --split-input-file --ttcore-register-device %s 2>&1 | FileCheck %s
// Negative unit tests for the ttnn.minimal_matmul_strided_reduce_scatter_async
// verifier. Verifier only inspects operand shapes, so plain (unencoded)
// tensors suffice.

// Bound multi-device semaphores must come as an exact pair.
module {
  func.func @wrong_semaphore_count(%input: tensor<32x128xbf16>, %weight: tensor<128x64xbf16>) -> tensor<32x64xbf16> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %sem = "ttnn.create_global_semaphore"(%device) <{core_range_set = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (3,0)>]>, initial_value = 0 : ui32}> : (!ttnn.device) -> !ttnn.global_semaphore
    // CHECK: error: 'ttnn.minimal_matmul_strided_reduce_scatter_async' op expects exactly two multi-device global semaphores, got 1
    %0 = "ttnn.minimal_matmul_strided_reduce_scatter_async"(%input, %weight, %sem, %device) <{
      cluster_axis = 1 : ui32,
      operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1, 0, 1>
    }> : (tensor<32x128xbf16>, tensor<128x64xbf16>, !ttnn.global_semaphore, !ttnn.device) -> tensor<32x64xbf16>
    return %0 : tensor<32x64xbf16>
  }
}

// -----

// The gated-residual fusion requires both addcmul operands together.
module {
  func.func @addcmul_only_one(%input: tensor<32x128xbf16>, %weight: tensor<128x64xbf16>, %res: tensor<32x64xbf16>) -> tensor<32x64xbf16> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.minimal_matmul_strided_reduce_scatter_async' op addcmul_input1 and addcmul_input2 must both be present or both absent
    %0 = "ttnn.minimal_matmul_strided_reduce_scatter_async"(%input, %weight, %res, %device) <{
      cluster_axis = 1 : ui32,
      operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 0, 0, 1>
    }> : (tensor<32x128xbf16>, tensor<128x64xbf16>, tensor<32x64xbf16>, !ttnn.device) -> tensor<32x64xbf16>
    return %0 : tensor<32x64xbf16>
  }
}

// -----

// Bias is row-broadcast: its last dim must equal the matmul width N.
module {
  func.func @bias_wrong_width(%input: tensor<32x128xbf16>, %weight: tensor<128x64xbf16>, %bias: tensor<1x32xbf16>) -> tensor<32x64xbf16> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.minimal_matmul_strided_reduce_scatter_async' op bias last dimension (32) must match weight's last dimension N (64)
    %0 = "ttnn.minimal_matmul_strided_reduce_scatter_async"(%input, %weight, %bias, %device) <{
      cluster_axis = 1 : ui32,
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 1>
    }> : (tensor<32x128xbf16>, tensor<128x64xbf16>, tensor<1x32xbf16>, !ttnn.device) -> tensor<32x64xbf16>
    return %0 : tensor<32x64xbf16>
  }
}

// -----

// Input must be at least rank 2 (a matrix).
module {
  func.func @input_rank_too_low(%input: tensor<128xbf16>, %weight: tensor<128x64xbf16>) -> tensor<32x64xbf16> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    // CHECK: error: 'ttnn.minimal_matmul_strided_reduce_scatter_async' op input tensor must have rank >= 2
    %0 = "ttnn.minimal_matmul_strided_reduce_scatter_async"(%input, %weight, %device) <{
      cluster_axis = 1 : ui32,
      operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 1>
    }> : (tensor<128xbf16>, tensor<128x64xbf16>, !ttnn.device) -> tensor<32x64xbf16>
    return %0 : tensor<32x64xbf16>
  }
}
