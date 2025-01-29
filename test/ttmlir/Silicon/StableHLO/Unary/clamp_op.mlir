// REQUIRES: stablehlo, num-chips-1 || num-chips-2
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_transpose attributes {} {
  func.func public @test_clamp_constant(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_clamp_constant
    // CHECK: ttnn.clamp
    // CHECK-SAME: {max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<64x128xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<64x128xf32>
    %0 = stablehlo.clamp %cst, %arg0, %cst_0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func public @test_clamp_tensor(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_clamp_tensor
    // CHECK: ttnn.empty
    // CHECK: %[[MAX:.*]] = "ttnn.maximum"
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    // CHECK: ttnn.empty
    // CHECK: "ttnn.minimum"(%[[MAX]]
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %0 = stablehlo.clamp %arg1, %arg0, %arg2 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
