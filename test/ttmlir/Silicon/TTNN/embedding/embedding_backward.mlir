// REQUIRES: num-chips-1 || num-chips-2
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module attributes {} {
  func.func @backward(%arg0: tensor<1x32xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<1x32x128xf32>) -> tensor<512x128xf32> {
    // CHECK: %{{[0-9]+}} = "ttnn.empty"
    %0 = tensor.empty() : tensor<512x128xf32>
    // CHECK: %{{[0-9]+}} = "ttnn.embedding_bw"
    %1 = "ttir.embedding_backward"(%arg0, %arg1, %arg2, %0) : (tensor<1x32xf32>, tensor<512x128xf32>, tensor<1x32x128xf32>, tensor<512x128xf32>) -> tensor<512x128xf32>
    return %1 : tensor<512x128xf32>
  }
}
