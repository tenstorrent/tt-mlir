// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module attributes {} {
  func.func @equal(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xf32> {
    %0 = ttir.empty() : tensor<13x31xf32>
    %1 = "ttir.eq"(%arg0, %arg1, %0) : (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xf32>
    // CHECK: "ttnn.eq"
    // CHECK-SAME: tensor<13x31xf32
    // CHECK-SAME: tensor<13x31xf32
    // CHECK-SAME: -> tensor<13x31xf32
    return %1 : tensor<13x31xf32>
  }
}
