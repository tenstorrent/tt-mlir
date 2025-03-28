// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for quantization operations

module attributes {} {
  func.func @test_rank_mismatch(%arg0: tensor<3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>> {
    // CHECK: error: 'stablehlo.uniform_quantize' op all non-scalar operands/results must have the same shape and base type
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>
    return %0 : tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>
  }
}

// -----

module attributes {} {
  func.func @test_shape_mismatch(%arg0: tensor<1x2x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>> {
    // CHECK: error: 'stablehlo.uniform_quantize' op all non-scalar operands/results must have the same shape and base type
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x2x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>
    return %0 : tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>
  }
}
