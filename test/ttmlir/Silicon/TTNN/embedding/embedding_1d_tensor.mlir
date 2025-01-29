// REQUIRES: num-chips-1 || num-chips-2
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module attributes {} {
  func.func @forward(%arg0: tensor<32xbf16>, %arg1: tensor<512x128xbf16>) -> tensor<32x128xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<32x128xbf16>
    // CHECK: %[[C:.*]] = "ttnn.embedding"[[C:.*]]
    %1 = "ttir.embedding"(%arg0, %arg1, %0) : (tensor<32xbf16>, tensor<512x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }
}
