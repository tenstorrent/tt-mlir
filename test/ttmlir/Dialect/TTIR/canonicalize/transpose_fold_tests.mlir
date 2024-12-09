// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
module {
  func.func @transpose_involution(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK-NOT: "ttir.transpose"
    %0 = tensor.empty() : tensor<128x64xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
    %2 = tensor.empty() : tensor<64x128xbf16>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = 1 : si32, dim1 = 0 : si32}> : (tensor<128x64xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %3 : tensor<64x128xbf16>
  }
}
