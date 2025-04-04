// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_exp attributes {} {
  func.func public @test_exp(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK-LABEL: func.func public @test_exp
    // CHECK: ttnn.exp
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: -> tensor<64x128xbf16,
    %0 = stablehlo.exponential %arg0 : tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}
