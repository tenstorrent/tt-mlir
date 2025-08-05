// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// UNSUPPORTED: Blackhole
// Embedding backward test fails on blackhole architecture due to the following metal issue:
// https://github.com/tenstorrent/tt-metal/issues/25260
module attributes {} {
  func.func @backward(%arg0: tensor<1x32xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<1x32x128xf32>) -> tensor<512x128xf32> {
    %0 = ttir.empty() : tensor<512x128xf32>
    // CHECK: %{{[0-9]+}} = "ttnn.embedding_bw"
    %1 = "ttir.embedding_backward"(%arg0, %arg1, %arg2, %0) : (tensor<1x32xf32>, tensor<512x128xf32>, tensor<1x32x128xf32>, tensor<512x128xf32>) -> tensor<512x128xf32>
    return %1 : tensor<512x128xf32>
  }
}
