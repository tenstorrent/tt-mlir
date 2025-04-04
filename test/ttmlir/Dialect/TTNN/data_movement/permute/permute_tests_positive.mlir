// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @permute_general(%arg0: tensor<8x32x64x128xbf16>) -> tensor<64x8x128x32xbf16> {
    %0 = ttir.empty() : tensor<64x8x128x32xbf16>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 2, 0, 3, 1>
    // CHECK-SAME: tensor<8x32x64x128xbf16
    // CHECK-SAME: tensor<64x8x128x32xbf16
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 2, 0, 3, 1>}> : (tensor<8x32x64x128xbf16>, tensor<64x8x128x32xbf16>) -> tensor<64x8x128x32xbf16>
    return %1 : tensor<64x8x128x32xbf16>
  }
}
