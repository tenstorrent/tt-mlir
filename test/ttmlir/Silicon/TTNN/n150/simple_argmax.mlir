// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module attributes {} {
  func.func public @argmax_2d(%arg0: tensor<64x64xf32>) -> tensor<64xi32> {
    // CHECK-LABEL: func.func public @argmax_2d(
    %0 = tensor.empty() : tensor<64xi32>
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, use_multicore = false}>
    // CHECK-SAME: tensor<1x1x64x64xbf16
    // CHECK-SAME: -> tensor<1x1x64x1xui32
    %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<64x64xf32>, tensor<64xi32>) -> tensor<64xi32>
    return %1 : tensor<64xi32>
  }

  func.func public @argmax_3d(%arg0: tensor<1x28x28xf32>) -> tensor<1x28xi32> {
    // CHECK-LABEL: func.func public @argmax_3d(
    %0 = tensor.empty() : tensor<1x28xi32>
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, use_multicore = false}>
    // CHECK-SAME: tensor<1x1x28x28xbf16
    // CHECK-SAME: -> tensor<1x1x28x1xui32
    %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x28x28xf32>, tensor<1x28xi32>) -> tensor<1x28xi32>
    return %1 : tensor<1x28xi32>
  }

  func.func public @argmax_4d(%arg0: tensor<1x1x128x64xf32>) -> tensor<1x1x128xi32> {
    // CHECK-LABEL: func.func public @argmax_4d(
    %0 = tensor.empty() : tensor<1x1x128xi32>
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: {dim = 3 : i32, use_multicore = false}>
    // CHECK-SAME: tensor<1x1x128x64xbf16
    // CHECK-SAME: -> tensor<1x1x128x1xui32
    %1 = "ttir.argmax"(%arg0, %0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x1x128x64xf32>, tensor<1x1x128xi32>) -> tensor<1x1x128xi32>
    return %1 : tensor<1x1x128xi32>
  }
}
