// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_constant attributes {} {
  func.func public @test_int32_scalar() -> tensor<i32> {
    // CHECK-LABEL: func.func public @test_int32_scalar
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 3 : i32
    // CHECK-SAME: -> tensor<f32
    // CHECK: ttnn.typecast
    // CHECK-SAME: -> tensor<si32
    %0 = stablehlo.constant dense<3> : tensor<i32>
    return %0 : tensor<i32>
  }

  func.func public @test_int32_scalar_empty() -> tensor<i32> {
    // CHECK-LABEL: func.func public @test_int32_scalar_empty
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<f32
    // CHECK: ttnn.typecast
    // CHECK-SAME: -> tensor<si32
    %0 = stablehlo.constant dense<0> : tensor<i32>
    return %0 : tensor<i32>
  }

  func.func public @test_int32_empty() -> tensor<64x128xi32> {
    // CHECK-LABEL: func.func public @test_int32_empty
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<64x128xf32
    // CHECK: ttnn.typecast
    // CHECK-SAME: -> tensor<64x128xsi32
    %0 = stablehlo.constant dense<0> : tensor<64x128xi32>
    return %0 : tensor<64x128xi32>
  }

  func.func public @test_int32_splat() -> tensor<64x128xi32> {
    // CHECK-LABEL: func.func public @test_int32_splat
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 3 : i32
    // CHECK-SAME: -> tensor<64x128xf32
    // CHECK: ttnn.typecast
    // CHECK-SAME: -> tensor<64x128xsi32
    %0 = stablehlo.constant dense<3> : tensor<64x128xi32>
    return %0 : tensor<64x128xi32>
  }
}
