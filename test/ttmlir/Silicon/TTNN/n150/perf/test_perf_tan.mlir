// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @tan(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: [[VAL0:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}})
  // CHECK: %{{[0-9]+}} = "ttnn.tan"(%arg0, [[VAL0]])
  %1 = "ttir.tan"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
