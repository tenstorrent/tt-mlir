// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_pow attributes {} {
  func.func public @test_power(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_power
    // CHECK: ttnn.pow
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %0 = stablehlo.power %arg0, %arg1 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func public @test_power_scalar(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_power
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<64x128xf32>
    // CHECK: ttnn.pow
    // CHECK-SAME: <{exponent = 2.000000e+00 : f32}>
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %0 = stablehlo.power %arg0, %cst : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
