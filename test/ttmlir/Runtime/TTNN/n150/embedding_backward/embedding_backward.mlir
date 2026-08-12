// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  func.func @embedding_backward(%arg0: tensor<1x32xi32>, %arg1: tensor<512x128xf32>, %arg2: tensor<1x32x128xf32>) -> tensor<512x128xf32> {
    // CHECK: "ttnn.embedding_bw"
    %0 = "ttir.embedding_backward"(%arg0, %arg1, %arg2) : (tensor<1x32xi32>, tensor<512x128xf32>, tensor<1x32x128xf32>) -> tensor<512x128xf32>
    return %0 : tensor<512x128xf32>
  }
}
