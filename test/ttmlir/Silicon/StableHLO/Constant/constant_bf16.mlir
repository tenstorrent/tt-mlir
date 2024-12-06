// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_constant attributes {} {
  func.func public @test_bfloat16_scalar() -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_bfloat16_scalar
    // CHECK: ttnn.full
    // CHECK-SAME: fillValue = 3.000000e+00 : f32
    // CHECK-SAME: -> tensor<1xbf16
    %0 = stablehlo.constant dense<3.0> : tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_bfloat16_scalar_empty() -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_bfloat16_scalar_empty
    // CHECK: ttnn.full
    // CHECK-SAME: -> tensor<1xbf16
    %0 = stablehlo.constant dense<0.0> : tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_bfloat16_empty() -> tensor<64x128xbf16> {
    // CHECK-LABEL: func.func public @test_bfloat16_empty
    // CHECK: ttnn.full
    // CHECK-SAME: -> tensor<64x128xbf16
    %0 = stablehlo.constant dense<0.0> : tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }

  func.func public @test_bfloat16_splat() -> tensor<64x128xbf16> {
    // CHECK-LABEL: func.func public @test_bfloat16_splat
    // CHECK: ttnn.full
    // CHECK-SAME: fillValue = 3.000000e+00 : f32
    // CHECK-SAME: -> tensor<64x128xbf16
    %0 = stablehlo.constant dense<3.0> : tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}
