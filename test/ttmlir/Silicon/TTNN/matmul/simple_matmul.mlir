// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
    %0 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    return %1 : tensor<64x96xbf16>
  }

  func.func @matmul_transpose_first(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<128x128xbf16> {
    %0 = tensor.empty() : tensor<128x128xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_a = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }

  func.func @matmul_transpose_second(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x64xbf16> {
    %0 = tensor.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_b = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
}
