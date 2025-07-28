// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Unit tests for ttir all_to_all op

// -----

module attributes {} {
  func.func public @all_to_all_split_dimension_range_1(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x32x32xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, concat_dim = 3 : si32, split_count = 2 : si32, split_dim = -1 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    return %1 : tensor<1x1x32x32xf32>
  }
}
// CHECK: error: 'ttir.all_to_all' op splitDim must be in the range [0, 3], got -1

// -----

module attributes {} {
  func.func public @all_to_all_split_dimension_range_2(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x32x32xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,  concat_dim = 3 : si32, split_count = 2 : si32, split_dim = 4 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    return %1 : tensor<1x1x32x32xf32>
  }
}
// CHECK: error: 'ttir.all_to_all' op splitDim must be in the range [0, 3], got 4

// -----

module attributes {} {
  func.func public @all_to_all_split_count_not_divisible(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x32x32xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,  concat_dim = 3 : si32, split_count = 3 : si32, split_dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    return %1 : tensor<1x1x32x32xf32>
  }
}
// CHECK: error: 'ttir.all_to_all' op splitDim size must be divisible by splitCount

// -----

module attributes {} {
  func.func public @all_to_all_concat_dimension_range_1(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x32x32xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,  concat_dim = -1 : si32, split_count = 2 : si32, split_dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    return %1 : tensor<1x1x32x32xf32>
  }
}
// CHECK: error: 'ttir.all_to_all' op concatDim must be in the range [0, 3], got -1

// -----

module attributes {} {
  func.func public @all_to_all_concat_dimension_range_2(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x32x32xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,  concat_dim = 4 : si32, split_count = 2 : si32, split_dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    return %1 : tensor<1x1x32x32xf32>
  }
}
// CHECK: error: 'ttir.all_to_all' op concatDim must be in the range [0, 3], got 4

// -----

module attributes {} {
  func.func public @all_to_all_split_count_range(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x32x32xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,  concat_dim = 3 : si32, split_count = 0 : si32, split_dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    return %1 : tensor<1x1x32x32xf32>
  }
}
// CHECK: error: 'ttir.all_to_all' op splitCount must be a positive integer


// -----

module attributes {} {
  func.func public @all_to_all_output_type_1(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x16x64xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x16x64xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,  concat_dim = 3 : si32, split_count = 2 : si32, split_dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x16x64xf32>) -> tensor<1x1x16x64xf32>
    return %1 : tensor<1x1x16x64xf32>
  }
}
// CHECK: error: 'ttir.all_to_all' op Output shape mismatch: expected = <1, 1, 32, 32> output = <1, 1, 16, 64>

// -----

module attributes {} {
  func.func public @all_to_all_output_type_2(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x32x32xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,  concat_dim = 2 : si32, split_count = 4 : si32, split_dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    return %1 : tensor<1x1x32x32xf32>
  }
}
// CHECK: error: 'ttir.all_to_all' op Output shape mismatch: expected = <1, 1, 128, 8> output = <1, 1, 32, 32>

// -----

module attributes {} {
  func.func public @all_to_all_output_type_3(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x32x32xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,  concat_dim = 3 : si32, split_count = 8 : si32, split_dim = 2 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    return %1 : tensor<1x1x32x32xf32>
  }
}
// CHECK: error: 'ttir.all_to_all' op Output shape mismatch: expected = <1, 1, 4, 256> output = <1, 1, 32, 32>

// -----

module attributes {} {
  func.func public @all_to_all_output_type_4(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xi32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x32x32xi32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,  concat_dim = 3 : si32, split_count = 2 : si32, split_dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xi32>) -> tensor<1x1x32x32xi32>
    return %1 : tensor<1x1x32x32xi32>
  }
}
// CHECK: error: 'ttir.all_to_all' op Input and output element types must match
