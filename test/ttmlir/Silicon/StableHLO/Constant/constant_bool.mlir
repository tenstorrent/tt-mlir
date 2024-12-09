// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_constant attributes {} {
  func.func public @test_boolean_scalar() -> tensor<i1> {
    // CHECK-LABEL: func.func public @test_boolean_scalar
    // CHECK: ttnn.full
    // CHECK-SAME: fillValue = 1.000000e+00 : f32
    // CHECK-SAME: -> tensor<1xbf16
    %0 = stablehlo.constant dense<true> : tensor<i1>
    return %0 : tensor<i1>
  }

  func.func public @test_boolean_scalar_empty() -> tensor<i1> {
    // CHECK-LABEL: func.func public @test_boolean_scalar_empty
    // CHECK: ttnn.full
    // CHECK-SAME: -> tensor<1xbf16
    %0 = stablehlo.constant dense<false> : tensor<i1>
    return %0 : tensor<i1>
  }

  func.func public @test_boolean_empty() -> tensor<64x128xi1> {
    // CHECK-LABEL: func.func public @test_boolean_empty
    // CHECK: ttnn.full
    // CHECK-SAME: -> tensor<64x128xbf16
    %0 = stablehlo.constant dense<false> : tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

  func.func public @test_boolean_splat() -> tensor<64x128xi1> {
    // CHECK-LABEL: func.func public @test_boolean_splat
    // CHECK: ttnn.full
    // CHECK-SAME: fillValue = 1.000000e+00 : f32
    // CHECK-SAME: -> tensor<64x128xbf16
    %0 = stablehlo.constant dense<true> : tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }
}
