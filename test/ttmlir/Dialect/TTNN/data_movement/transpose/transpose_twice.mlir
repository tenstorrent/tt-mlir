// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x16x32x64xbf16>) -> tensor<1x32x64x16xbf16> {
    %0 = ttir.empty() : tensor<1x64x32x16xbf16>
    // Due to the canonicalization of the `ttir.transpose` op, we are expecting `ttnn.permute` op.
    // CHECK: = "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -3 : si32, dim1 = -1 : si32}> : (tensor<1x16x32x64xbf16>, tensor<1x64x32x16xbf16>) -> tensor<1x64x32x16xbf16>
    %2 = ttir.empty() : tensor<1x32x64x16xbf16>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x64x32x16xbf16>, tensor<1x32x64x16xbf16>) -> tensor<1x32x64x16xbf16>
    return %3 : tensor<1x32x64x16xbf16>
  }
}
