// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

// Test pow operation with broadcasting
func.func @pow_with_broadcast(%arg0: tensor<64x1xf32>, %arg1: tensor<1x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.pow"(%arg0, %arg1, %0) : (tensor<64x1xf32>, tensor<1x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: "ttnn.pow"
  // CHECK-SAME: tensor<64x128xf32
  // CHECK-SAME: tensor<64x128xf32
  // CHECK-SAME: -> tensor<64x128xf32
  return %1 : tensor<64x128xf32>
}
