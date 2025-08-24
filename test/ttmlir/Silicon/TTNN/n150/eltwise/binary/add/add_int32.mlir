// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @addint32(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
  %0 = ttir.empty() : tensor<64x128xi32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
  // CHECK: "ttnn.add"
  // CHECK-SAME: tensor<64x128xsi32
  // CHECK-SAME: tensor<64x128xsi32
  // CHECK-SAME: -> tensor<64x128xsi32
  return %1 : tensor<64x128xi32>
}
