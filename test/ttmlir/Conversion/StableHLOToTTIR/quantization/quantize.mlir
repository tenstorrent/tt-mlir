// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_quantize attributes {} {
  func.func @test_uniform_quantize(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>> {
    // CHECK-LABEL: func.func @test_uniform_quantize
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<1x3x224x224xi32>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.quantize"(%arg0, %[[EMPTY]]) {scale = 2.000000e-02 : f32, zero_point = 0 : i32}
    // CHECK-SAME: (tensor<1x3x224x224xf32>, [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %0 : tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>
  }

  func.func @test_per_channel_quantize(%arg0: tensor<3x3x7x7xf32>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02,0.01,0.005}>> {
    // CHECK-LABEL: func.func @test_per_channel_quantize
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<3x3x7x7xi32>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.quantize"(%arg0, %[[EMPTY]])
    // CHECK-SAME: {axis = 0 : i32, scale = [2.000000e-02 : f32, 1.000000e-02 : f32, 5.000000e-03 : f32], zero_point = [0 : i32, 0 : i32, 0 : i32]}
    // CHECK-SAME: (tensor<3x3x7x7xf32>, [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<3x3x7x7xf32>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02,0.01,0.005}>>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %0 : tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02,0.01,0.005}>>
  }


}
