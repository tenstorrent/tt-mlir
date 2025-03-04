// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module attributes {} {
  func.func public @argmax_2d(%arg0: tensor<64x64xf32>) -> tensor<64x1xi32> {
    // CHECK-LABEL: func.func public @argmax_2d(
    %0 = tensor.empty() : tensor<64x1xi32>
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, use_multicore = false}>
    // CHECK-SAME: tensor<1x1x64x64xbf16
    // CHECK-SAME: tensor<1x1x64x1xsi32
    // CHECK-SAME: -> tensor<1x1x64x1xsi32
    %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<64x64xf32>, tensor<64x1xi32>) -> tensor<64x1xi32>
    return %1 : tensor<64x1xi32>
  }

  func.func public @argmax_3d(%arg0: tensor<128x28x28xf32>) -> tensor<128x28xi32> {
    // CHECK-LABEL: func.func public @argmax_3d(
    %0 = tensor.empty() : tensor<128x28xi32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, use_multicore = false}>
    // CHECK-SAME: tensor<1x128x28x28xbf16
    // CHECK-SAME: tensor<1x128x28x1xsi32
    // CHECK-SAME: -> tensor<1x128x28x1xsi32
    %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<128x28x28xf32>, tensor<128x28xi32>) -> tensor<128x28xi32>
    return %1 : tensor<128x28xi32>
  }

  func.func public @argmax_4d(%arg0: tensor<4x8x128x64xf32>) -> tensor<4x8x128xi32> {
    // CHECK-LABEL: func.func public @argmax_4d(
    %0 = tensor.empty() : tensor<4x8x128xi32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, use_multicore = false}>
    // CHECK-SAME: tensor<4x8x128x64xbf16
    // CHECK-SAME: tensor<4x8x128x1xsi32
    // CHECK-SAME: -> tensor<4x8x128x1xsi32
    %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<4x8x128x64xf32>, tensor<4x8x128xi32>) -> tensor<4x8x128xi32>
    return %1 : tensor<4x8x128xi32>
  }
}
