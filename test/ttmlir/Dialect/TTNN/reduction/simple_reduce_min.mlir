// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module attributes {} {
  func.func public @test_reduce_min_4to3dim(%arg0: tensor<128x64x32x4xf32>) -> tensor<128x32x4xf32> {
    // CHECK-LABEL: func.func public @test_reduce_min_4to3dim
    %0 = ttir.empty() : tensor<128x32x4xf32>
    // CHECK: %[[MIN:[0-9]+]] = "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x64x32x4xf32,
    // CHECK-SAME: -> tensor<128x32x4xf32,
    %1 = "ttir.min"(%arg0, %0) <{dim_arg = [1: i32], keep_dim = false}> : (tensor<128x64x32x4xf32>, tensor<128x32x4xf32>) -> tensor<128x32x4xf32>
    return %1 : tensor<128x32x4xf32>
  }

  func.func public @test_reduce_min_4to0dim(%arg0: tensor<128x64x32x32xbf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_min_4to0dim
    %0 = ttir.empty() : tensor<bf16>
    // CHECK-NOT: dim_arg = [1 : i32]
    // CHECK: %[[MIN:[0-9]+]] = "ttnn.min"
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x64x32x32xbf16,
    // CHECK-SAME: -> tensor<bf16,
    %1 = "ttir.min"(%arg0, %0) <{dim_arg = [0 : i32, 1 : i32, 2 : i32, 3 : i32], keep_dim = false}> : (tensor<128x64x32x32xbf16>, tensor<bf16>) -> tensor<bf16>
    return %1 : tensor<bf16>
  }

  func.func public @test_reduce_min_3to2dim(%arg0: tensor<128x32x4xf32>) -> tensor<128x4xf32> {
    // CHECK-LABEL: func.func public @test_reduce_min_3to2dim
    %0 = ttir.empty() : tensor<128x4xf32>
    // CHECK: %[[MIN:[0-9]+]] = "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x32x4xf32,
    // CHECK-SAME: -> tensor<128x4xf32,
    %1 = "ttir.min"(%arg0, %0) <{dim_arg = [1: i32], keep_dim = false}> : (tensor<128x32x4xf32>, tensor<128x4xf32>) -> tensor<128x4xf32>
    return %1 : tensor<128x4xf32>
  }

  func.func public @test_reduce_min_3to0dim(%arg0: tensor<128x32x64xbf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_min_3to0dim
    %0 = ttir.empty() : tensor<bf16>
    // CHECK-NOT: dim_arg = [1 : i32]
    // CHECK: %[[MIN:[0-9]+]] = "ttnn.min"
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x32x64xbf16,
    // CHECK-SAME: -> tensor<bf16,
    %1 = "ttir.min"(%arg0, %0) <{dim_arg = [0 : i32, 1 : i32, 2 : i32], keep_dim = false}> : (tensor<128x32x64xbf16>, tensor<bf16>) -> tensor<bf16>
    return %1 : tensor<bf16>
  }

  func.func public @test_reduce_min_1to0dim(%arg0: tensor<128xbf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_min_1to0dim
    %0 = ttir.empty() : tensor<bf16>
    // CHECK-NOT: dim_arg = [0 : i32]
    // CHECK-NOT: ttnn.reshape
    // CHECK: %[[MIN:[0-9]+]] = "ttnn.min"
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128xbf16,
    // CHECK-SAME: -> tensor<bf16,
    %1 = "ttir.min"(%arg0, %0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<128xbf16>, tensor<bf16>) -> tensor<bf16>
    return %1 : tensor<bf16>
  }
}
