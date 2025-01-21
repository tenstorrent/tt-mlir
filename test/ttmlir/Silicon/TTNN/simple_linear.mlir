// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @simple_linear_with_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<64x64xbf16
    %0 = tensor.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
}
