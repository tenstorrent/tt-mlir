// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @simple_linear_with_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = ttir.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.matmul"
    // CHECK: tensor<64x128xbf16
    // CHECK: tensor<128x64xbf16
    // CHECK: tensor<64x64xbf16
    // CHECK: tensor<64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }


  func.func @linear_transpose_lhs(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %bias: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %0 = ttir.empty() : tensor<128x128xbf16>
    // CHECK: "ttnn.matmul"
    // CHECK: transpose_a = true
    // CHECK: transpose_b = false
    // CHECK: tensor<64x128xbf16
    // CHECK: tensor<64x128xbf16
    // CHECK: tensor<128x128xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) <{transpose_a = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }

  func.func @linear_transpose_second(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = ttir.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.matmul"
    // CHECK: transpose_a = false
    // CHECK: transpose_b = true
    // CHECK: tensor<64x128xbf16
    // CHECK: tensor<64x128xbf16
    // CHECK: tensor<64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) <{transpose_b = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
}
