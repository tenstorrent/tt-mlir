// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for ttnn collective_permute op

// Verify lowering of ttir collective_permute to ttnn ops
module attributes {} {
  // CHECK-LABEL: collective_permute_valid_case
  func.func public @collective_permute_valid_case(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = "ttir.collective_permute"(%arg0) <{source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttnn.point_to_point"
    // CHECK-SAME: receiver_coord = array<i64: 0, 1>, sender_coord = array<i64: 0, 0>
    // CHECK: "ttnn.point_to_point"
    // CHECK-SAME: receiver_coord = array<i64: 0, 0>, sender_coord = array<i64: 0, 1>
    return %0 : tensor<1x1x8192x512xf32>
  }
}

// -----

// Verify op folding for zero sized source-target pair
module attributes {} {
  // CHECK-LABEL: collective_permute_zero_sized_pair
  func.func public @collective_permute_zero_sized_pair(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = "ttir.collective_permute"(%arg0) <{source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK-NOT: "ttnn.point_to_point"
    return %0 : tensor<1x1x8192x512xf32>
  }
}

// -----

// Verify collective_permute op with self-mapped pairs
module attributes {} {
  // CHECK-LABEL: collective_permute_self_mapped_pair
  func.func public @collective_permute_self_mapped_pair(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = "ttir.collective_permute"(%arg0) <{source_target_pairs = dense<[[0, 0], [1, 1]]> : tensor<2x2xi64>}> : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttnn.assign"
    // CHECK-NOT: "ttnn.point_to_point"
    return %0 : tensor<1x1x8192x512xf32>
  }
}
