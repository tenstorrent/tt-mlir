// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround %s | FileCheck %s

module  {
  func.func @embedding_weight_6D(%input: tensor<2x4xui32>, %weight: tensor<1x1x1x1x10x10xbf16>) -> tensor<2x4x10xbf16> {
    // CHECK-LABEL: func.func @embedding_weight_6D
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 10 : i32, 10 : i32]
    // CHECK: "ttnn.embedding"
    %result = "ttnn.embedding"(%input, %weight) : (tensor<2x4xui32>, tensor<1x1x1x1x10x10xbf16>) -> tensor<2x4x10xbf16>
    return %result : tensor<2x4x10xbf16>
  }
}
