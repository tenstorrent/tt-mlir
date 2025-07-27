// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_constant attributes {} {
  func.func public @test_uint64_scalar() -> tensor<ui32> {
    // CHECK-LABEL: func.func public @test_uint64_scalar
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 3 : i32
    // CHECK-SAME: -> tensor<ui32
    %0 = stablehlo.constant dense<3> : tensor<ui32>
    return %0 : tensor<ui32>
  }

  func.func public @test_uint64_scalar_empty() -> tensor<ui32> {
    // CHECK-LABEL: func.func public @test_uint64_scalar_empty
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<ui32
    %0 = stablehlo.constant dense<0> : tensor<ui32>
    return %0 : tensor<ui32>
  }

  func.func public @test_uint64_empty() -> tensor<64x128xui32> {
    // CHECK-LABEL: func.func public @test_uint64_empty
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<64x128xui32
    %0 = stablehlo.constant dense<0> : tensor<64x128xui32>
    return %0 : tensor<64x128xui32>
  }

  func.func public @test_uint64_splat() -> tensor<64x128xui32> {
    // CHECK-LABEL: func.func public @test_uint64_splat
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 3 : i32
    // CHECK-SAME: -> tensor<64x128xui32
    %0 = stablehlo.constant dense<3> : tensor<64x128xui32>
    return %0 : tensor<64x128xui32>
  }
}
