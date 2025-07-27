// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_constant attributes {} {
  func.func public @test_uint16_scalar() -> tensor<ui16> {
    // CHECK-LABEL: func.func public @test_uint16_scalar
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 3 : i32
    // CHECK-SAME: -> tensor<ui16
    %0 = stablehlo.constant dense<3> : tensor<ui16>
    return %0 : tensor<ui16>
  }

  func.func public @test_uint16_scalar_empty() -> tensor<ui16> {
    // CHECK-LABEL: func.func public @test_uint16_scalar_empty
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<ui16
    %0 = stablehlo.constant dense<0> : tensor<ui16>
    return %0 : tensor<ui16>
  }

  func.func public @test_uint16_empty() -> tensor<64x128xui16> {
    // CHECK-LABEL: func.func public @test_uint16_empty
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<64x128xui16
    %0 = stablehlo.constant dense<0> : tensor<64x128xui16>
    return %0 : tensor<64x128xui16>
  }

  func.func public @test_uint16_splat() -> tensor<64x128xui16> {
    // CHECK-LABEL: func.func public @test_uint16_splat
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 3 : i32
    // CHECK-SAME: -> tensor<64x128xui16
    %0 = stablehlo.constant dense<3> : tensor<64x128xui16>
    return %0 : tensor<64x128xui16>
  }
}
