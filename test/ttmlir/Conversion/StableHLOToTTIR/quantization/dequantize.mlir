// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_dequantize attributes {} {
  func.func @dequantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>) -> tensor<1x3x224x224xf32> {
    // CHECK-LABEL: func.func @dequantize_per_tensor_scale_per_tensor_zp(
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<1x3x224x224xf32>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.dequantize"(%arg0, %[[EMPTY]])
    // CHECK-SAME: (tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>, tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    %0 = stablehlo.uniform_dequantize %arg0 : (tensor<1x3x224x224x!quant.uniform<i32:f32, 0.02>>) -> tensor<1x3x224x224xf32>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %0 : tensor<1x3x224x224xf32>
  }

  func.func @dequantize_per_axis_scale_per_tensor_zp(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02,0.01,0.005}>>) -> tensor<3x3x7x7xf32> {
    // CHECK-LABEL: func.func @dequantize_per_axis_scale_per_tensor_zp(
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<3x3x7x7xf32>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.dequantize"(%arg0, %[[EMPTY]])
    // CHECK-SAME: (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>>, tensor<3x3x7x7xf32>) -> tensor<3x3x7x7xf32>
    %0 = stablehlo.uniform_dequantize %arg0 : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02,0.01,0.005}>>) -> tensor<3x3x7x7xf32>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %0 : tensor<3x3x7x7xf32>
  }

  func.func @dequantize_per_axis_scale_per_axis_zp(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02:10,0.01:20,0.005:30}>>) -> tensor<3x3x7x7xf32> {
    // CHECK-LABEL: func.func @dequantize_per_axis_scale_per_axis_zp(
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<3x3x7x7xf32>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.dequantize"(%arg0, %[[EMPTY]])
    // CHECK-SAME: (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02:10,1.000000e-02:20,5.000000e-03:30}>>, tensor<3x3x7x7xf32>) -> tensor<3x3x7x7xf32>
    %0 = stablehlo.uniform_dequantize %arg0 : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {0.02:10,0.01:20,0.005:30}>>) -> tensor<3x3x7x7xf32>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %0 : tensor<3x3x7x7xf32>
  }
}
