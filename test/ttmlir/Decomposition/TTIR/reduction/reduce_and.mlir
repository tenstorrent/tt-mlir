// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module attributes {} {
  func.func public @test_reduce_and_4to3dim(%arg0: tensor<128x10x32x4xbf16>, %arg1: tensor<1xbf16>) -> tensor<128x10x32xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_and_4to3dim
    // CHECK: %[[PROD:[0-9]+]] = "ttir.prod"
    // CHECK-SAME: dim_arg = [3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xbf16>
    // CHECK-SAME: -> tensor<128x10x32xbf16>
    // CHECK: return %[[PROD]]
    %0 = tensor.empty() : tensor<128x10x32xbf16>
    %1 = "ttir.reduce_and"(%arg0, %0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<128x10x32x4xbf16>, tensor<128x10x32xbf16>) -> tensor<128x10x32xbf16>
    return %1 : tensor<128x10x32xbf16>
  }

  func.func public @test_reduce_and_3to2dim(%arg0: tensor<128x10x4xbf16>, %arg1: tensor<1xbf16>) -> tensor<128x4xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_and_3to2dim
    // CHECK: %[[PROD:[0-9]+]] = "ttir.prod"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xbf16>
    // CHECK-SAME: -> tensor<128x4xbf16>
    // CHECK: return %[[PROD]]
    %0 = tensor.empty() : tensor<128x4xbf16>
    %1 = "ttir.reduce_and"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x10x4xbf16>, tensor<128x4xbf16>) -> tensor<128x4xbf16>
    return %1 : tensor<128x4xbf16>
  }

  func.func public @test_reduce_and_2to1dim(%arg0: tensor<128x10xbf16>, %arg1: tensor<1xbf16>) -> tensor<10xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_and_2to1dim
    // CHECK: %[[PROD:[0-9]+]] = "ttir.prod"
    // CHECK-SAME: dim_arg = [0 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10xbf16>
    // CHECK-SAME: -> tensor<10xbf16>
    // CHECK: return %[[PROD]]
    %0 = tensor.empty() : tensor<10xbf16>
    %1 = "ttir.reduce_and"(%arg0, %0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<128x10xbf16>, tensor<10xbf16>) -> tensor<10xbf16>
    return %1 : tensor<10xbf16>
  }
}
