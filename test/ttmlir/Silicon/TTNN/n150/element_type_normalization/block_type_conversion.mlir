// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-bfp8-conversion=true" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module  {
  // CHECK-LABEL: @forward(
  // CHECK-SAME: tensor<32x32xbf16
  // CHECK-SAME: tensor<32x32xbf16
  // CHECK-SAME: -> tensor<32x32xbf16
  func.func @forward(%arg0 : tensor<32x32xbf16>, %arg1 : tensor<32x32xbf16>) ->tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttnn.add"
    // CHECK-SAME: tensor<32x32x!ttcore.tile<32x32, bfp_bf8
    // CHECK-SAME: tensor<32x32x!ttcore.tile<32x32, bfp_bf8
    // CHECK-SAME: -> tensor<32x32x!ttcore.tile<32x32, bfp_bf8
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
