// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Unit tests for ttir collective_permute op

// -----

module attributes {} {
  func.func public @collective_permute_invalid_source_target_pair_rank(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x8192x512xf32>
    %1 = "ttir.collective_permute"(%arg0, %0) <{source_target_pairs = dense<[0]> : tensor<1xi64>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    return %1 : tensor<1x1x8192x512xf32>
  }
}
// CHECK: error: 'ttir.collective_permute' op The rank of source target pairs must be 2, got rank = 1

// -----

module attributes {} {
  func.func public @collective_permute_invalid_duplicate_sources(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x8192x512xf32>
    %1 = "ttir.collective_permute"(%arg0, %0) <{source_target_pairs = dense<[[0, 1], [0, 2]]> : tensor<2x2xi64>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    return %1 : tensor<1x1x8192x512xf32>
  }
}
// CHECK: error: 'ttir.collective_permute' op There are duplicate 'src' or 'dest' devices in source target pairs


// -----

module attributes {} {
  func.func public @collective_permute_invalid_duplicate_targets(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x8192x512xf32>
    %1 = "ttir.collective_permute"(%arg0, %0) <{source_target_pairs = dense<[[0, 2], [1, 2]]> : tensor<2x2xi64>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    return %1 : tensor<1x1x8192x512xf32>
  }
}
// CHECK: error: 'ttir.collective_permute' op There are duplicate 'src' or 'dest' devices in source target pairs
