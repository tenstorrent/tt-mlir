// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_transpose attributes {} {
  func.func public @test_transpose(%arg0: tensor<64x128xf32>) -> tensor<128x64xf32> {
    // CHECK-LABEL: func.func public @test_transpose
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 1, 0>
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<128x64xf32,
    %0 = stablehlo.transpose %arg0, dims = [1,0] : (tensor<64x128xf32>) -> tensor<128x64xf32>
    return %0 : tensor<128x64xf32>
  }
}
