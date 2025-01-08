// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @tanh(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %0 = tensor.empty() : tensor<64x128xbf16>
  // CHECK: [[VAL0:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}})
  // CHECK: %{{[0-9]+}} = "ttnn.tanh"(%arg0, [[VAL0]])
  %1 = "ttir.tanh"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}
