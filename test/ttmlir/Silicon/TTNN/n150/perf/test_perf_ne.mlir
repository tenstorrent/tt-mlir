// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module attributes {} {
  func.func @not_equal(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xf32> {
    %0 = tensor.empty() : tensor<13x31xf32>
    %1 = "ttir.ne"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xf32>
    // CHECK: "ttnn.ne"
    // CHECK-SAME: tensor<13x31xf32
    // CHECK-SAME: tensor<13x31xf32
    // CHECK-SAME: -> tensor<13x31xf32
    return %1 : tensor<13x31xf32>
  }
}
