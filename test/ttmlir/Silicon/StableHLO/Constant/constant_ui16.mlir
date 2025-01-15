// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_constant attributes {} {
  func.func public @test_uint16_scalar() -> tensor<ui16> {
    // CHECK-LABEL: func.func public @test_uint16_scalar
    // CHECK: ttnn.full
    // CHECK-SAME: fillValue = 3.000000e+00 : f32
    // CHECK-SAME: -> tensor<1xui16
    %0 = stablehlo.constant dense<3> : tensor<ui16>
    return %0 : tensor<ui16>
  }

  func.func public @test_uint16_scalar_empty() -> tensor<ui16> {
    // CHECK-LABEL: func.func public @test_uint16_scalar_empty
    // CHECK: ttnn.full
    // CHECK-SAME: -> tensor<1xui16
    %0 = stablehlo.constant dense<0> : tensor<ui16>
    return %0 : tensor<ui16>
  }

  func.func public @test_uint16_empty() -> tensor<64x128xui16> {
    // CHECK-LABEL: func.func public @test_uint16_empty
    // CHECK: ttnn.full
    // CHECK-SAME: -> tensor<64x128xui16
    %0 = stablehlo.constant dense<0> : tensor<64x128xui16>
    return %0 : tensor<64x128xui16>
  }

  func.func public @test_uint16_splat() -> tensor<64x128xui16> {
    // CHECK-LABEL: func.func public @test_uint16_splat
    // CHECK: ttnn.full
    // CHECK-SAME: fillValue = 3.000000e+00 : f32
    // CHECK-SAME: -> tensor<64x128xui16
    %0 = stablehlo.constant dense<3> : tensor<64x128xui16>
    return %0 : tensor<64x128xui16>
  }
}
