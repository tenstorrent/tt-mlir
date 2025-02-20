// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @power(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  %1 = "ttir.power"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[RETURN_VALUE:[0-9]+]] = "ttnn.pow"(%arg0, %arg1)
  // CHECK-SAME: (tensor<32x32xf32, {{.*}}>, tensor<32x32xf32, {{.*}}>)
  // CHECK-SAME: -> tensor<32x32xf32, {{.*}}>
  return %1 : tensor<32x32xf32>
  // CHECK: return %[[RETURN_VALUE]] : tensor<32x32xf32, {{.*}}
}
