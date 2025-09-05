// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @requantize_per_tensor_scales_per_tensor_zps(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>> {
    // CHECK-LABEL: func.func @requantize_per_tensor_scales_per_tensor_zps(
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 2.000000e-01 : f32
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 1.000000e-01 : f32
    // CHECK: "ttnn.requantize"
    // CHECK-SAME: {output_dtype = #ttcore.supportedDataTypes<si32>}
    // CHECK-SAME: tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>,
    // CHECK-SAME: -> tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>,
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
  }
  func.func @requantize_per_axis_scales_per_tensor_zps(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01,4.000000e-01,6.000000e-01}>> {
    // CHECK-LABEL: func.func @requantize_per_axis_scales_per_tensor_zps(
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01,4.000000e-01,6.000000e-01}>>
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 1.000000e-01, 2.000000e-01, 3.000000e-01
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 2.000000e-01, 4.000000e-01, 6.000000e-01
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 0
    // CHECK: "ttnn.requantize"
    // CHECK-SAME: <{axis = 1 : i32, output_dtype = #ttcore.supportedDataTypes<si32>}
    // CHECK-SAME: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>,
    // CHECK-SAME: -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01,4.000000e-01,6.000000e-01}>,
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>>, tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01,4.000000e-01,6.000000e-01}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01,4.000000e-01,6.000000e-01}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01,4.000000e-01,6.000000e-01}>>
  }
  func.func @requantize_per_axis_scales_per_axis_zps(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01:10,4.000000e-01:20,6.000000e-01:30}>> {
    // CHECK-LABEL: func.func @requantize_per_axis_scales_per_axis_zps(
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01:10,4.000000e-01:20,6.000000e-01:30}>>
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 2.000000e-01, 4.000000e-01, 6.000000e-01
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 10, 20, 30
    // CHECK-SAME: -> tensor<3xsi32,
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 1.000000e-01, 2.000000e-01, 3.000000e-01
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.requantize"
    // CHECK-SAME: <{axis = 1 : i32, output_dtype = #ttcore.supportedDataTypes<si32>}
    // CHECK-SAME: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>,
    // CHECK-SAME: -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01:10,4.000000e-01:20,6.000000e-01:30}>,
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>, tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01:10,4.000000e-01:20,6.000000e-01:30}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01:10,4.000000e-01:20,6.000000e-01:30}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {2.000000e-01:10,4.000000e-01:20,6.000000e-01:30}>>
  }
}
