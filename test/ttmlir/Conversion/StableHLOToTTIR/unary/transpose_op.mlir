// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_transpose attributes {} {
  func.func public @test_transpose(%arg0: tensor<64x128xf32>) -> tensor<128x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1,0] : (tensor<64x128xf32>) -> tensor<128x64xf32>
    // CHECK: ttir.empty
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 1, 0>
    return %0 : tensor<128x64xf32>
  }
}
