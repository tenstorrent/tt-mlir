// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @addint32(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
  %0 = tensor.empty() : tensor<64x128xi32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
  // CHECK: "ttnn.add"
  // CHECK-SAME: tensor<64x128xsi32
  // CHECK-SAME: tensor<64x128xsi32
  // CHECK-SAME: -> tensor<64x128xsi32
  return %1 : tensor<64x128xi32>
}
