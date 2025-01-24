// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module attributes {} {
  func.func public @test_reduce_prod_4to3dim(%arg0: tensor<128x10x32x4xf32>) -> tensor<128x32x4xf32> {
    // CHECK-LABEL: func.func public @test_reduce_prod_4to3dim
    %0 = tensor.empty() : tensor<128x32x4xf32>
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = false
    // CHECK-SAME: dim_arg = 1
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10x32x4xf32,
    // CHECK-SAME: -> tensor<128x32x4xf32,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [1: i32], keep_dim = false}> : (tensor<128x10x32x4xf32>, tensor<128x32x4xf32>) -> tensor<128x32x4xf32>
    return %1 : tensor<128x32x4xf32>
  }

  func.func public @test_reduce_prod_4to0dim(%arg0: tensor<128x10x32x4xbf16>) -> tensor<1xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_4to0dim
    %0 = tensor.empty() : tensor<1xbf16>
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = true
    // CHECK-SAME: dim_arg = 0
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10x32x4xbf16,
    // CHECK-SAME: -> tensor<1xbf16,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [0 : i32, 1 : i32, 2 : i32, 3 : i32], keep_dim = false}> : (tensor<128x10x32x4xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

  func.func public @test_reduce_prod_3to2dim(%arg0: tensor<128x10x4xf32>) -> tensor<128x4xf32> {
    // CHECK-LABEL: func.func public @test_reduce_prod_3to2dim
    %0 = tensor.empty() : tensor<128x4xf32>
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = false
    // CHECK-SAME: dim_arg = 1
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10x4xf32,
    // CHECK-SAME: -> tensor<128x4xf32,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [1: i32], keep_dim = false}> : (tensor<128x10x4xf32>, tensor<128x4xf32>) -> tensor<128x4xf32>
    return %1 : tensor<128x4xf32>
  }

  func.func public @test_reduce_prod_3to0dim(%arg0: tensor<128x10x4xbf16>) -> tensor<1xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_3to0dim
    %0 = tensor.empty() : tensor<1xbf16>
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = true
    // CHECK-SAME: dim_arg = 0
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10x4xbf16,
    // CHECK-SAME: -> tensor<1xbf16,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [0 : i32, 1 : i32, 2 : i32], keep_dim = false}> : (tensor<128x10x4xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

  func.func public @test_reduce_prod_1to0dim(%arg0: tensor<128xbf16>) -> tensor<1xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_1to0dim
    %0 = tensor.empty() : tensor<1xbf16>
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = true
    // CHECK-SAME: dim_arg = 0
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128xbf16,
    // CHECK-SAME: -> tensor<1xbf16,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<128xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }
}
