// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
func.func @div(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %0 = ttir.empty() : tensor<64x128xbf16>
  %1 = "ttir.div"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  // CHECK: "ttnn.divide"
  // CHECK-SAME: tensor<64x128xbf16
  // CHECK-SAME: tensor<64x128xbf16
  // CHECK-SAME: -> tensor<64x128xbf16
  return %1 : tensor<64x128xbf16>
}
