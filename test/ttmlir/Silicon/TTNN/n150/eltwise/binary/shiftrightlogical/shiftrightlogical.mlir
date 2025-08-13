// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @shift_right_logical(%arg0: tensor<5xbf16>, %arg1: tensor<5xbf16>) -> tensor<5xbf16> {
  %0 = ttir.empty() : tensor<5xbf16>
  %1 = "ttir.shift_right_logical"(%arg0, %arg1, %0) : (tensor<5xbf16>, tensor<5xbf16>, tensor<5xbf16>) -> tensor<5xbf16>
  // CHECK: "ttnn.logical_right_shift"
  // CHECK-SAME: tensor<5xbf16
  // CHECK-SAME: tensor<5xbf16
  // CHECK-SAME: -> tensor<5xbf16
  return %1 : tensor<5xbf16>
}
