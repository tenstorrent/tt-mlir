// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s -o %t.mlir --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%"
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @test_clamp attributes {} {
  func.func public @test_clamp_constant(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_clamp_constant(
    // CHECK: ttnn.clamp_scalar
    // CHECK-SAME: {max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<64x128xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<64x128xf32>
    %0 = stablehlo.clamp %cst, %arg0, %cst_0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func public @test_clamp_indirect_constant_reshape(%arg0: tensor<1x16xbf16>) -> tensor<1x16xbf16> {
    // CHECK-LABEL: func.func public @test_clamp_indirect_constant_reshape
    %cst = arith.constant dense<3.0> : tensor<1xf64>
    %cst_0 = arith.constant dense<6> : tensor<1xi64>
    %0 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1xbf16>) -> tensor<bf16>
    %2 = stablehlo.convert %cst_0 : (tensor<1xi64>) -> tensor<1xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1xbf16>) -> tensor<bf16>
    // CHECK: ttnn.clamp_scalar
    // CHECK-SAME: {max = 6.000000e+00 : f32, min = 3.000000e+00 : f32}
    // CHECK-SAME: tensor<1x16xbf16,
    // CHECK-SAME: -> tensor<1x16xbf16,
    %4 = stablehlo.clamp %1, %arg0, %3 : (tensor<bf16>, tensor<1x16xbf16>, tensor<bf16>) -> tensor<1x16xbf16>
    return %4 : tensor<1x16xbf16>
  }

  func.func public @test_clamp_indirect_constant_broadcast(%arg0: tensor<1x32xbf16>) -> (tensor<1x32xbf16>) {
    // CHECK-LABEL: func.func public @test_clamp_indirect_constant_broadcast
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %cst_0 = stablehlo.constant dense<5.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<1x32xbf16>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<1x32xbf16>
    // CHECK: ttnn.clamp_scalar
    // CHECK-SAME: {max = 5.000000e+00 : f32, min = 2.000000e+00 : f32}
    // CHECK-SAME: tensor<1x32xbf16,
    // CHECK-SAME: -> tensor<1x32xbf16,
    %2 = stablehlo.clamp %0, %arg0, %1 : tensor<1x32xbf16>
    return %2 : tensor<1x32xbf16>
  }
}
