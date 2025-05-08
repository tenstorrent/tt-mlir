// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_quantize {
  func.func @quantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>> {
    // CHECK-LABEL: func.func @quantize_per_tensor_scale_per_tensor_zp(
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
    // CHECK: %[[RET:[0-9]+]] = "ttir.quantize"(%arg0, %[[EMPTY]]) : (tensor<1x3x224x224xf32>, tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
    // CHECK: return %[[RET]] : tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
    return %0 : tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
  }

  func.func @quantize_per_axis_scale_per_tensor_zp(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>> {
    // CHECK-LABEL: func.func @quantize_per_axis_scale_per_tensor_zp(
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>
    // CHECK: %[[RET:[0-9]+]] = "ttir.quantize"(%arg0, %[[EMPTY]]) : (tensor<1x3x224x224xf32>, tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>
    // CHECK: return %[[RET]] : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>
    return %0 : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>
  }

  func.func @quantize_per_axis_scale_per_axis_zp(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>> {
    // CHECK-LABEL: func.func @quantize_per_axis_scale_per_axis_zp(
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
    // CHECK: %[[RET:[0-9]+]] = "ttir.quantize"(%arg0, %[[EMPTY]]) : (tensor<1x3x224x224xf32>, tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
    // CHECK: return %[[RET]] : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
    return %0 : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
  }
}
