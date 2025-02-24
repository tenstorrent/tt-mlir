// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module attributes {} {
  func.func public @argmax_2d(%arg0: tensor<64x64xf32>) -> tensor<64x1xi32> {
    // CHECK-LABEL: func.func public @argmax_2d(
    %0 = tensor.empty() : tensor<64x1xi32>
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 1 : i32, use_multicore = false}>
    // CHECK-SAME: tensor<64x64xf32
    // CHECK-SAME: tensor<64x1xsi32
    // CHECK-SAME: -> tensor<64x1xsi32
    %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<64x64xf32>, tensor<64x1xi32>) -> tensor<64x1xi32>
    return %1 : tensor<64x1xi32>
  }

  func.func public @argmax_3d(%arg0: tensor<128x28x28xf32>) -> tensor<128x28xi32> {
    // CHECK-LABEL: func.func public @argmax_3d(
    %0 = tensor.empty() : tensor<128x28xi32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttnn.argmax"
    // CHECK-SAME: {dim = 2 : i32, use_multicore = false}>
    // CHECK-SAME: tensor<128x28x28xf32
    // CHECK: %[[TYPECAST:[0-9]+]] = "ttnn.typecast"(%[[ARGMAX]])
    // CHECK-SAME: -> tensor<128x28x1xui32
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"(%[[TYPECAST]])
    // CHECK-SAME: <{shape = [128 : i32, 28 : i32]}>
    // CHECK-SAME: -> tensor<128x28xui32
    // CHECK: "ttnn.typecast"(%[[RESHAPE]])
    // CHECK-SAME: -> tensor<128x28xsi32
    %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<128x28x28xf32>, tensor<128x28xi32>) -> tensor<128x28xi32>
    return %1 : tensor<128x28xi32>
  }

  func.func public @argmax_4d(%arg0: tensor<4x8x128x64xf32>) -> tensor<4x8x128xi32> {
    // CHECK-LABEL: func.func public @argmax_4d(
    %0 = tensor.empty() : tensor<4x8x128xi32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, use_multicore = false}>
    // CHECK-SAME: tensor<4x8x128x64xf32
    // CHECK: %[[TYPECAST:[0-9]+]] = "ttnn.typecast"(%[[ARGMAX]])
    // CHECK-SAME: -> tensor<4x8x128x1xui32
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"(%[[TYPECAST]])
    // CHECK-SAME: <{shape = [4 : i32, 8 : i32, 128 : i32]}>
    // CHECK-SAME: -> tensor<4x8x128xui32
    // CHECK: "ttnn.typecast"(%[[RESHAPE]])
    // CHECK-SAME: -> tensor<4x8x128xsi32
    %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<4x8x128x64xf32>, tensor<4x8x128xi32>) -> tensor<4x8x128xi32>
    return %1 : tensor<4x8x128xi32>
  }
}
