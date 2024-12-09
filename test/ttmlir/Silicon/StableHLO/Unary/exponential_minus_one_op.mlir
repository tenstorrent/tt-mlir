// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_expm1 attributes {} {
  func.func public @test_expm1(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_expm1
    // CHECK: ttnn.empty
    // CHECK: ttnn.expm1
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %0 = stablehlo.exponential_minus_one %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
