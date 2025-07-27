// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func public @test_reduce_add_4to2dim(%arg0: tensor<128x10x32x4xf32>) -> tensor<128x32xf32> {
    // CHECK-LABEL: @test_reduce_add_4to2dim(
    %0 = ttir.empty() : tensor<128x32xf32>
    // CHECK: %[[PROD:[0-9]+]] = "ttir.prod"(%arg0, %{{[0-9]+}})
    // CHECK-SAME: <{dim_arg = [3 : i32], keep_dim = false}>
    // CHECK-SAME: (tensor<128x10x32x4xf32>, tensor<128x10x32xf32>) -> tensor<128x10x32xf32>
    // CHECK: %{{[0-9]+}} = "ttir.prod"(%[[PROD]], %{{[0-9]+}})
    // CHECK-SAME: <{dim_arg = [1 : i32], keep_dim = false}>
    // CHECK-SAME: (tensor<128x10x32xf32>, tensor<128x32xf32>) -> tensor<128x32xf32>
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [1 : i32, 3 : i32], keep_dim = false}> : (tensor<128x10x32x4xf32>, tensor<128x32xf32>) -> tensor<128x32xf32>
    return %1 : tensor<128x32xf32>
  }

  func.func public @test_reduce_add_4to2dim_keepdim(%arg0: tensor<128x10x32x4xf32>) -> tensor<128x1x32x1xf32> {
    // CHECK-LABEL: @test_reduce_add_4to2dim_keepdim
    %0 = ttir.empty() : tensor<128x1x32x1xf32>
    // CHECK: %[[PROD:[0-9]+]] = "ttir.prod"(%arg0, %{{[0-9]+}})
    // CHECK-SAME: <{dim_arg = [3 : i32], keep_dim = true}>
    // CHECK-SAME: (tensor<128x10x32x4xf32>, tensor<128x10x32x1xf32>) -> tensor<128x10x32x1xf32>
    // CHECK: %{{[0-9]+}} = "ttir.prod"(%[[PROD]], %{{[0-9]+}})
    // CHECK-SAME: <{dim_arg = [1 : i32], keep_dim = true}>
    // CHECK-SAME: (tensor<128x10x32x1xf32>, tensor<128x1x32x1xf32>) -> tensor<128x1x32x1xf32>
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [1 : i32, 3 : i32], keep_dim = true}> : (tensor<128x10x32x4xf32>, tensor<128x1x32x1xf32>) -> tensor<128x1x32x1xf32>
    return %1 : tensor<128x1x32x1xf32>
  }
}
