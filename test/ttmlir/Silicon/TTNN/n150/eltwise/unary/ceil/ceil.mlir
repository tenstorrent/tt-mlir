// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @ceil(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  %1 = "ttir.ceil"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[RETURN_VALUE:[0-9]+]] = "ttnn.ceil"(%arg0)
  // CHECK-SAME: (tensor<32x32xf32, {{.*}}>)
  // CHECK-SAME: -> tensor<32x32xf32, {{.*}}>
  return %1 : tensor<32x32xf32>
  // CHECK: return %[[RETURN_VALUE]] : tensor<32x32xf32, {{.*}}>
}
