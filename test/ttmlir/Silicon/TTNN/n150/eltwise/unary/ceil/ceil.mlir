// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @ceil(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = ttir.empty() : tensor<32x32xbf16>
  %1 = "ttir.ceil"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // CHECK: "ttnn.ceil"
  // CHECK-SAME: tensor<32x32xbf16
  // CHECK-SAME: -> tensor<32x32xbf16
  return %1 : tensor<32x32xbf16>
}
