// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=2,1,1" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true
// REQUIRES: multi-chip
func.func @multiply(%arg0: tensor<8x64x128xf32>, %arg1: tensor<8x64x128xf32>) -> tensor<8x64x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<8x64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.multiply"[[C:.*]]
  %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<8x64x128xf32>, tensor<8x64x128xf32>, tensor<8x64x128xf32>) -> tensor<8x64x128xf32>
  return %1 : tensor<8x64x128xf32>
}
