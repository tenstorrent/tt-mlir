// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Verify that the parsing fails if the tensor has less than 2 dimensions
module attributes {} {
  func.func public @test_batch_norm_rank_too_low(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<2xf32> {
    %0 = ttir.empty() : tensor<2xf32>
    %1 = "ttir.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 0 : i32, epsilon = 0.000000e+00 : f32}> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    // CHECK: error: 'ttir.batch_norm' op input tensor must have rank between 2 and 5
    return %1 : tensor<2xf32>
  }
}

// -----

// Verify that the parsing fails if the tensor has more than 5 dimensions
module attributes {} {
  func.func public @test_batch_norm_rank_too_high(%arg0: tensor<2x2x2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<2x2x2x2x2x2xf32> {
    %0 = ttir.empty() : tensor<2x2x2x2x2x2xf32>
    %1 = "ttir.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 1 : i32, epsilon = 0.000000e+00 : f32}> : (tensor<2x2x2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2x2x2xf32>) -> tensor<2x2x2x2x2x2xf32>
    // CHECK: error: 'ttir.batch_norm' op input tensor must have rank between 2 and 5
    return %1 : tensor<2x2x2x2x2x2xf32>
  }
}

// -----

// Verify that the parsing fails if dimension is out of bounds
module attributes {} {
  func.func public @test_batch_norm_dimension_out_of_bounds(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<2x2x2xf32> {
    %0 = ttir.empty() : tensor<2x2x2xf32>
    %1 = "ttir.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 5 : i32, epsilon = 0.000000e+00 : f32}> : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    // CHECK: error: 'ttir.batch_norm' op dimension attribute must be within input rank bounds
    return %1 : tensor<2x2x2xf32>
  }
}
