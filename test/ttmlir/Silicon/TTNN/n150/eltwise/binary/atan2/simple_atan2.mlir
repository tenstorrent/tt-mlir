// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  func.func @atan2(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.atan2"(%arg0, %arg1, %0) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: "ttnn.atan2"
    // CHECK-SAME: tensor<32x32xf32
    // CHECK-SAME: tensor<32x32xf32
    // CHECK-SAME: -> tensor<32x32xf32
    return %1 : tensor<32x32xf32>
  }
}
