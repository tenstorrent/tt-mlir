// REQUIRES: stablehlo, num-chips-1 || num-chips-2
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_constant attributes {} {
  func.func public @test_int32_scalar() -> tensor<i32> {
    // CHECK-LABEL: func.func public @test_int32_scalar
    // CHECK: ttnn.full
    // CHECK-SAME: fillValue = 3.000000e+00 : f32
    // CHECK-SAME: -> tensor<1xi32
    %0 = stablehlo.constant dense<3> : tensor<i32>
    return %0 : tensor<i32>
  }

  func.func public @test_int32_scalar_empty() -> tensor<i32> {
    // CHECK-LABEL: func.func public @test_int32_scalar_empty
    // CHECK: ttnn.full
    // CHECK-SAME: -> tensor<1xi32
    %0 = stablehlo.constant dense<0> : tensor<i32>
    return %0 : tensor<i32>
  }

  func.func public @test_int32_empty() -> tensor<64x128xi32> {
    // CHECK-LABEL: func.func public @test_int32_empty
    // CHECK: ttnn.full
    // CHECK-SAME: -> tensor<64x128xi32
    %0 = stablehlo.constant dense<0> : tensor<64x128xi32>
    return %0 : tensor<64x128xi32>
  }

  func.func public @test_int32_splat() -> tensor<64x128xi32> {
    // CHECK-LABEL: func.func public @test_int32_splat
    // CHECK: ttnn.full
    // CHECK-SAME: fillValue = 3.000000e+00 : f32
    // CHECK-SAME: -> tensor<64x128xi32
    %0 = stablehlo.constant dense<3> : tensor<64x128xi32>
    return %0 : tensor<64x128xi32>
  }
}
