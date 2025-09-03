// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
func.func @logicalleftshift(%arg0: tensor<5xui32>, %arg1: tensor<5xui32>) -> tensor<5xui32> {
  %0 = ttir.empty() : tensor<5xui32>
  %1 = "ttir.logical_left_shift"(%arg0, %arg1, %0) : (tensor<5xui32>, tensor<5xui32>, tensor<5xui32>) -> tensor<5xui32>
  // CHECK: "ttnn.logical_left_shift"
  // CHECK-SAME: tensor<5xui32
  // CHECK-SAME: tensor<5xui32
  // CHECK-SAME: -> tensor<5xui32
  return %1 : tensor<5xui32>
}
