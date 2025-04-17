// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @quantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>> {
    // CHECK-LABEL: func.func @quantize_per_tensor_scale_per_tensor_zp(
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<1.000000e-01> : tensor<1xf32,
    // CHECK-SAME: -> tensor<1xf32,
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<0> : tensor<1xi32,
    // CHECK-SAME: -> tensor<1xsi32,
    // CHECK: "ttnn.quantize"
    // CHECK-SAME: {output_dtype = #tt.supportedDataTypes<si32>}
    // CHECK-SAME: tensor<1x3x320x320xf32
    // CHECK-SAME: -> tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>,
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
  }
  func.func @quantize_per_axis_scale_per_tensor_zp(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>> {
    // CHECK-LABEL: func.func @quantize_per_axis_scale_per_tensor_zp(
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>>
    // CHECK: "ttnn.constant"()
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 1.000000e-01, 2.000000e-01, 3.000000e-01
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.full"
    // CHECK-SAME: fillValue = 0.000000e+00 : f32
    // CHECK-SAME: -> tensor<3xui32,
    // CHECK: "ttnn.quantize"
    // CHECK-SAME: {axis = 1 : i32, output_dtype = #tt.supportedDataTypes<si32>}
    // CHECK-SAME: tensor<1x3x320x320xf32
    // CHECK-SAME: -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>,
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>>
  }
  func.func @quantize_per_axis_scale_per_axis_zp(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>> {
    // CHECK-LABEL: func.func @quantize_per_axis_scale_per_axis_zp(
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>
    // CHECK: "ttnn.constant"()
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 1.000000e-01, 2.000000e-01, 3.000000e-01
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.constant"()
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 10, 20, 30
    // CHECK-SAME: -> tensor<3xsi32,
    // CHECK: "ttnn.quantize"
    // CHECK-SAME: <{axis = 1 : i32, output_dtype = #tt.supportedDataTypes<si32>}>
    // CHECK-SAME: tensor<1x3x320x320xf32
    // CHECK-SAME: -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>,
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>
  }
}
