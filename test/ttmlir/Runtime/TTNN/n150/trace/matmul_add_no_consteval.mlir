// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false enable-trace=true" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @matmul_with_bias(%arg0: tensor<64x32xbf16>, %arg1: tensor<32x64xbf16>, %arg2: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: ttnn.trace
    %0 = ttir.empty() : tensor<64x64xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x32xbf16>, tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = ttir.empty() : tensor<64x64xbf16>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}
