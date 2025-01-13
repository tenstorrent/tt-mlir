// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
func.func @logical_and(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: {{.*}} = "ttnn.empty"{{.*}}
  %1 = "ttir.logical_and"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.logical_and"
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: tensor<64x128xf32,
  return %1 : tensor<64x128xf32>
}
