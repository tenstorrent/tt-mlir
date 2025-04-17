// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @dequantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>) -> tensor<1x3x320x320xf32> {
    // CHECK-LABEL: func.func @dequantize_per_tensor_scale_per_tensor_zp(
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<1.000000e-01> : tensor<1xf32,
    // CHECK-SAME: -> tensor<1xf32,
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<0> : tensor<1xi32,
    // CHECK-SAME: -> tensor<1xsi32,
    // CHECK: "ttnn.dequantize"
    // CHECK-SAME: {output_dtype = #tt.supportedDataTypes<f32>}
    // CHECK-SAME: tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>,
    // CHECK-SAME: -> tensor<1x3x320x320xf32,
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
  func.func @dequantize_per_channel_scale_per_tensor_zp(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>>) -> tensor<1x3x320x320xf32> {
    // CHECK-LABEL: func.func @dequantize_per_channel_scale_per_tensor_zp(
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 1.000000e-01, 2.000000e-01, 3.000000e-01
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.full"
    // CHECK-SAME: fillValue = 0.000000e+00 : f32
    // CHECK-SAME: -> tensor<3xui32,
    // CHECK: "ttnn.dequantize"
    // CHECK-SAME: <{axis = 1 : i32, output_dtype = #tt.supportedDataTypes<f32>}>
    // CHECK-SAME: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>,
    // CHECK-SAME: -> tensor<1x3x320x320xf32,
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
  func.func @dequantize_per_channel_scale_per_channel_zp(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>) -> tensor<1x3x320x320xf32> {
    // CHECK-LABEL: func.func @dequantize_per_channel_scale_per_channel_zp(
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 1.000000e-01, 2.000000e-01, 3.000000e-01
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<
    // CHECK-SAME: 10, 20, 30
    // CHECK-SAME: -> tensor<3xsi32,
    // CHECK: "ttnn.dequantize"
    // CHECK-SAME: <{axis = 1 : i32, output_dtype = #tt.supportedDataTypes<f32>}>
    // CHECK-SAME: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>,
    // CHECK-SAME: -> tensor<1x3x320x320xf32,
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}
