// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @simple_linear_with_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128x64xbf16
    // CHECK-SAME: tensor<64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    %0 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }

  func.func @simple_linear_with_2d_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.add"
    %0 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }

  func.func @linear_transpose_lhs_1d_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %bias: tensor<128xbf16>) -> tensor<128x128xbf16> {
    // CHECK: "ttnn.linear"
    // CHECK-SAME: transpose_a = true
    // CHECK-SAME: transpose_b = false
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128x128xbf16
    %0 = "ttir.linear"(%arg0, %arg1, %bias) <{transpose_a = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<128xbf16>) -> tensor<128x128xbf16>
    return %0 : tensor<128x128xbf16>
  }

  func.func @linear_transpose_lhs_2d_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %bias: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: transpose_a = true
    // CHECK-SAME: transpose_b = false
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128x128xbf16
    // CHECK: "ttnn.add"
    %0 = "ttir.linear"(%arg0, %arg1, %bias) <{transpose_a = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %0 : tensor<128x128xbf16>
  }

  func.func @linear_transpose_second_1d_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %bias: tensor<64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: "ttnn.linear"
    // CHECK-SAME: transpose_a = false
    // CHECK-SAME: transpose_b = true
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    %0 = "ttir.linear"(%arg0, %arg1, %bias) <{transpose_b = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }

  func.func @linear_transpose_second_2d_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: transpose_a = false
    // CHECK-SAME: transpose_b = true
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x64xbf16
    %0 = "ttir.linear"(%arg0, %arg1, %bias) <{transpose_b = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }
}
