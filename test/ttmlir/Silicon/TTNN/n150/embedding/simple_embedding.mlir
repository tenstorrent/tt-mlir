// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xf32>, %arg1: tensor<512x128xf32>) -> tensor<32x32x128xf32> {
    // CHECK: = "ttnn.empty"
    %0 = tensor.empty() : tensor<32x32x128xf32>
    // CHECK: = "ttnn.embedding"
    %1 = "ttir.embedding"(%arg0, %arg1, %0) : (tensor<32x32xf32>, tensor<512x128xf32>, tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
    return %1 : tensor<32x32x128xf32>
  }
}
