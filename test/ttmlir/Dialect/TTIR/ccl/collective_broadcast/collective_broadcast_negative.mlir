// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Unit tests for ttir collective_broadcast op

// -----

module attributes {} {
  func.func public @collective_broadcast_invalid_output_type(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x4096x512xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x4096x512xf32>
    %1 = "ttir.collective_broadcast"(%arg0, %0) <{replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x4096x512xf32>) -> tensor<1x1x4096x512xf32>
    return %1 : tensor<1x1x4096x512xf32>
  }
}
// CHECK: error: 'ttir.collective_broadcast' op input and output must have the same type

// -----

module attributes {} {
  func.func public @collective_broadcast_invalid_result_type(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x4096x512xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x8192x512xf32>
    %1 = "ttir.collective_broadcast"(%arg0, %0) <{replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x512xf32>
    return %1 : tensor<1x1x4096x512xf32>
  }
}
// CHECK: error: 'ttir.collective_broadcast' op output and result must have the same type

// -----

module attributes {} {
  func.func public @collective_broadcast_invalid_replica_groups_shape(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x8192x512xf32>
    %1 = "ttir.collective_broadcast"(%arg0, %0) <{replica_groups = dense<[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]> : tensor<2x2x2xi64>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    return %1 : tensor<1x1x8192x512xf32>
  }
}
// CHECK: error: 'ttir.collective_broadcast' op replica_groups must be a 2D array

// -----

module attributes {} {
  func.func public @collective_broadcast_negative_device_id(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x8192x512xf32>
    %1 = "ttir.collective_broadcast"(%arg0, %0) <{replica_groups = dense<[[0, 1, 2, -1], [4, 5, 6, -1]]> : tensor<2x4xi64>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    return %1 : tensor<1x1x8192x512xf32>
  }
}
// CHECK: error: 'ttir.collective_broadcast' op replica_groups values must be positive

// -----

module attributes {} {
  func.func public @collective_broadcast_duplicated_device_id(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x8192x512xf32>
    %1 = "ttir.collective_broadcast"(%arg0, %0) <{replica_groups = dense<[[0, 1, 2, 3], [0, 5, 6, 7]]> : tensor<2x4xi64>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    return %1 : tensor<1x1x8192x512xf32>
  }
}
// CHECK: error: 'ttir.collective_broadcast' op replica_groups must not contain duplicate values

// -----

module attributes {} {
  func.func public @collective_broadcast_duplicated_device_id(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x8192x512xf32>
    %1 = "ttir.collective_broadcast"(%arg0, %0) <{replica_groups = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi64>}> : (tensor<1x1x8192x512xf32>, tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    return %1 : tensor<1x1x8192x512xf32>
  }
}
// CHECK: error: 'ttir.collective_broadcast' op replica_groups values must be in the range [0, 7], got 8
