// REQUIRES: num-chips-1 || num-chips-2
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
func.func @logical_not(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: {{.*}} = "ttnn.empty"{{.*}}
  %1 = "ttir.logical_not"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.logical_not"
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: tensor<64x128xf32,
  return %1 : tensor<64x128xf32>
}
