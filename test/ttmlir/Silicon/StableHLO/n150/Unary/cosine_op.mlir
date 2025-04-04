// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_cosine attributes {} {
  func.func public @test_cosine(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK-LABEL: func.func public @test_cosine
    // CHECK: ttnn.cos
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: -> tensor<64x128xbf16,
    %0 = stablehlo.cosine %arg0 : tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}
