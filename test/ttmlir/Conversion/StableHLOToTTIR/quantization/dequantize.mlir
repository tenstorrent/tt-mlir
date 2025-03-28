// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @jit_dequantize attributes {} {
  func.func @test_uniform_dequantize(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>) -> tensor<1x3x224x224xf32> {
    // CHECK-LABEL: func.func @test_uniform_dequantize
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<1x3x224x224xf32>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.dequantize"(%arg0, %[[EMPTY]]) {scale = 2.000000e-02 : f32, zero_point = 0 : i32}
    // CHECK-SAME: (tensor<1x3x224x224xi32>, [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.uniform_dequantize %arg0 : (tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>) -> tensor<1x3x224x224xf32>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %0 : tensor<1x3x224x224xf32>
  }

  func.func @test_per_channel_dequantize(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02,0.01,0.005}>>) -> tensor<3x3x7x7xf32> {
    // CHECK-LABEL: func.func @test_per_channel_dequantize
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<3x3x7x7xf32>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.dequantize"(%arg0, %[[EMPTY]])
    // CHECK-SAME: {axis = 0 : i32, scale = [2.000000e-02 : f32, 1.000000e-02 : f32, 5.000000e-03 : f32], zero_point = [0 : i32, 0 : i32, 0 : i32]}
    // CHECK-SAME: (tensor<3x3x7x7xi32>, [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.uniform_dequantize %arg0 : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02,0.01,0.005}>>) -> tensor<3x3x7x7xf32>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %0 : tensor<3x3x7x7xf32>
  }

  func.func @test_rank_mismatch(%arg0: tensor<3x224x224x!quant.uniform<i32:f32, 0.02>>) -> tensor<1x3x224x224xf32> {
    // expected-error@+1 {'stablehlo.uniform_dequantize' op all non-scalar operands/results must have the same shape and base type}
    %0 = stablehlo.uniform_dequantize %arg0 : (tensor<3x224x224x!quant.uniform<i32:f32, 0.02>>) -> tensor<1x3x224x224xf32>
    return %0 : tensor<1x3x224x224xf32>
  }

  func.func @test_shape_mismatch(%arg0: tensor<1x2x224x224x!quant.uniform<i32:f32, 0.02>>) -> tensor<1x3x224x224xf32> {
    // expected-error@+1 {'stablehlo.uniform_dequantize' op all non-scalar operands/results must have the same shape and base type}
    %0 = stablehlo.uniform_dequantize %arg0 : (tensor<1x2x224x224x!quant.uniform<i32:f32, 0.02>>) -> tensor<1x3x224x224xf32>
    return %0 : tensor<1x3x224x224xf32>
  }
}
