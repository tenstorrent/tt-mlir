// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func public @argmax_2d(%arg0: tensor<64x64xf32>) -> tensor<64x1xi32> {
    // CHECK-LABEL: func.func public @argmax_2d(
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 1 : i32, keep_dim = true, use_multicore = true}>
    // CHECK-SAME: tensor<64x64xf32
    // CHECK-SAME: -> tensor<64x1xui32
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<64x64xf32>) -> tensor<64x1xi32>
    return %1 : tensor<64x1xi32>
  }

  func.func public @argmax_3d(%arg0: tensor<128x28x28xf32>) -> tensor<128x28xi32> {
    // CHECK-LABEL: func.func public @argmax_3d(
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 1>
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 2 : i32, keep_dim = false, use_multicore = true}>
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x28x28xf32>) -> tensor<128x28xi32>
    return %1 : tensor<128x28xi32>
  }

  func.func public @argmax_3d_last_dim(%arg0: tensor<128x28x28xf32>) -> tensor<128x28xi32> {
    // CHECK-LABEL: func.func public @argmax_3d_last_dim(
    // CHECK-NOT: "ttnn.permute"
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 2 : i32, keep_dim = false, use_multicore = true}>
    // CHECK-SAME: tensor<128x28x28xf32
    // CHECK-SAME: -> tensor<128x28xui32
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<128x28x28xf32>) -> tensor<128x28xi32>
    return %1 : tensor<128x28xi32>
  }

  func.func public @argmax_4d(%arg0: tensor<4x8x128x64xf32>) -> tensor<4x8x128xi32> {
    // CHECK-LABEL: func.func public @argmax_4d(
    // CHECK-NOT: "ttnn.permute"
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, keep_dim = false, use_multicore = true}>
    // CHECK-SAME: tensor<4x8x128x64xf32
    // CHECK-SAME: -> tensor<4x8x128xui32
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<4x8x128x64xf32>) -> tensor<4x8x128xi32>
    return %1 : tensor<4x8x128xi32>
  }
}
