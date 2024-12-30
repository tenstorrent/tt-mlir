// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
func.func @neg_dim_five(%arg0: tensor<4x2x32x32xbf16>) -> tensor<1x4x2x32x32xbf16> {
  %0 = tensor.empty() : tensor<1x4x2x32x32xbf16>
  // CHECK: "ttnn.reshape"
  // CHECK-SAME: shape = [1 : i32, 4 : i32, 2 : i32, 32 : i32, 32 : i32]}
  // CHECK-SAME: tensor<4x2x32x32xbf16
  // CHECK-SAME: -> tensor<1x4x2x32x32xbf16
  %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = -5 : si32}> : (tensor<4x2x32x32xbf16>, tensor<1x4x2x32x32xbf16>) -> tensor<1x4x2x32x32xbf16>
  return %1 : tensor<1x4x2x32x32xbf16>
}
