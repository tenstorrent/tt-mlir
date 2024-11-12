// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module attributes {} {
  func.func @backward(%arg0: tensor<1x32xbf16>, %arg1: tensor<512x128xbf16>, %arg2: tensor<1x32x128xbf16>) -> tensor<512x128xbf16> {
    // CHECK: %{{[0-9]+}} = "ttnn.empty"
    %0 = tensor.empty() : tensor<512x128xbf16>
    // CHECK: %{{[0-9]+}} = "ttnn.embedding_bw"
    %1 = "ttir.embedding_backward"(%arg0, %arg1, %arg2, %0) : (tensor<1x32xbf16>, tensor<512x128xbf16>, tensor<1x32x128xbf16>, tensor<512x128xbf16>) -> tensor<512x128xbf16>
    return %1 : tensor<512x128xbf16>
  }
}
