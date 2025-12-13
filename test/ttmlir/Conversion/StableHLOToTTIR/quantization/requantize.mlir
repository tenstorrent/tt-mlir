// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_requantize {
  func.func @test_uniform_requantize(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>> {
    // CHECK-LABEL: func.func @test_uniform_requantize(
    // CHECK: %[[RET:[0-9]+]] = "ttir.requantize"(%arg0)
    // CHECK-SAME: (tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>
    // CHECK: return %[[RET]] : tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>
    return %0 : tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>
  }

  func.func @test_per_axis_requantize(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02,5.000000e-03,2.500000e-03}>> {
    // CHECK-LABEL: func.func @test_per_axis_requantize(
    // CHECK: %[[RET:[0-9]+]] = "ttir.requantize"(%arg0)
    // CHECK-SAME: (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02,5.000000e-03,2.500000e-03}>>
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02,5.000000e-03,2.500000e-03}>>
    // CHECK: return %[[RET]] : tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02,5.000000e-03,2.500000e-03}>>
    return %0 : tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02,5.000000e-03,2.500000e-03}>>
  }
}
