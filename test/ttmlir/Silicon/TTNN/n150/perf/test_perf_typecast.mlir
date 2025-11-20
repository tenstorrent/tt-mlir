
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
func.func @typecast(%arg0: tensor<64x128xf32>) -> tensor<64x128xbf16> {
  // CHECK: = "ttnn.typecast"
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: tensor<64x128xbf16,
  %1 = "ttir.typecast"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}
