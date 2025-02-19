// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1xbf16> {
    %0 = tensor.empty() : tensor<512x1xbf16>
    // CHECK: %[[C:.*]] = "ttnn.mean"[[C:.*]]
    %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<512x1024xbf16>, tensor<512x1xbf16>) -> tensor<512x1xbf16>
    return %1 : tensor<512x1xbf16>
  }

  func.func @test_reduce_mean_multi_dim(%arg0: tensor<128x32x10x4xbf16>) -> tensor<128x1x1x1xbf16> {
    // CHECK-LABEL: @test_reduce_mean_multi_dim(
    %0 = tensor.empty() : tensor<128x1x1x1xbf16>
    // CHECK: "ttnn.mean"(%arg0)
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x32x10x4xbf16
    // CHECK-SAME: -> tensor<128x1x1x1xbf16
    %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [1: i32, 2: i32, 3: i32], keep_dim = true}> : (tensor<128x32x10x4xbf16>, tensor<128x1x1x1xbf16>) -> tensor<128x1x1x1xbf16>
    return %1 : tensor<128x1x1x1xbf16>
  }
}
