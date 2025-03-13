// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,1" %s | FileCheck %s
// Unit tests for ttnn collective_permute op

// -----

// Verify lowering of ttir collective_permute to ttnn ops
module attributes {} {
  func.func public @collective_permute_invalid_duplicate_sources(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = tensor.empty() : tensor<1x1x8192x512xf32>
    %1 = "ttir.collective_permute"(%arg0, %0) <{source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    return %1 : tensor<1x1x8192x512xf32>
  }
}
// CHECK: "ttnn.collective_permute"
