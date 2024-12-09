// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module attributes {} {
  func.func @less_than(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty
    // CHECK-SAME: [[TENSOR:tensor<13x31xf32,]]
    %0 = tensor.empty() : tensor<13x31xf32>
    // CHECK: %[[C:.*]] = "ttnn.lt"
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: -> [[TENSOR]]
    %1 = "ttir.lt"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xf32>
    return %1 : tensor<13x31xf32>
  }
}
