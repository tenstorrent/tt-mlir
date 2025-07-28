// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  func.func @logical_not(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.logical_not"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: "ttnn.logical_not"
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: -> tensor<64x128xf32
    return %1 : tensor<64x128xf32>
  }
}
