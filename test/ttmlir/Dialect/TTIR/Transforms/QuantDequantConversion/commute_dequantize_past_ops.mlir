// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @commute_dequantize_past_avgpool2d(%arg0: tensor<1x112x112x64xf32>) -> tensor<1x114x114x64xf32> {
    // It is not safe to commute past avgpool2d, so a avgpool2d -> quantize -> dequantize is performed.
    // CHECK-LABEL: @commute_dequantize_past_avgpool2d
    // CHECK: ttnn.quantize
    // CHECK: ttnn.dequantize
    // CHECK: ttnn.avg_pool2d
    // CHECK: ttnn.quantize
    // CHECK: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0) : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.dequantize"(%1) : (tensor<1x112x112x64x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x112x112x64xf32>
    %5 = "ttir.avg_pool2d"(%3) <{kernel = array<i32: 1, 1>, stride = array<i32: 1, 1>, dilation = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, ceil_mode = false}> : (tensor<1x112x112x64xf32>) -> tensor<1x114x114x64xf32>
    return %5 : tensor<1x114x114x64xf32>
  }
  func.func @commute_dequantize_past_two_maxpool2d(%arg0: tensor<1x112x112x64xf32>) -> tensor<1x28x28x64xf32> {
    // It is safe to commute past maxpool2d.
    // CHECK-LABEL: @commute_dequantize_past_two_maxpool2d
    // CHECK: ttnn.quantize
    // CHECK: ttnn.max_pool2d
    // CHECK: ttnn.max_pool2d
    // CHECK: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0) : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.dequantize"(%1) : (tensor<1x112x112x64x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x112x112x64xf32>
    %5 = "ttir.max_pool2d"(%3) <{kernel = array<i32: 3, 3>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, ceil_mode = false}> : (tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32>
    %7 = "ttir.max_pool2d"(%5) <{kernel = array<i32: 3, 3>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, ceil_mode = false}> : (tensor<1x56x56x64xf32>) -> tensor<1x28x28x64xf32>
    return %7 : tensor<1x28x28x64xf32>
  }
  func.func @commute_dequantize_past_maximum(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    // It is not safe to commute past maximum, so commute dequantize down with q -> dq sandwich.
    // CHECK-LABEL: @commute_dequantize_past_maximum
    // CHECK: ttnn.quantize
    // CHECK: ttnn.dequantize
    // CHECK: ttnn.relu
    // CHECK: ttnn.quantize
    // CHECK: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.dequantize"(%1) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112xf32>
    %4 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x64x112x112xf32>}> : () -> tensor<1x64x112x112xf32>
    %6 = "ttir.maximum"(%3, %4) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %6 : tensor<1x64x112x112xf32>
  }
  func.func @commute_dequantize_past_add_successful(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    // The quantized types of the operands both align, so simply commute dequantize down.
    // CHECK-LABEL: @commute_dequantize_past_add_successful
    // CHECK: ttnn.quantize
    // CHECK: ttnn.quantize
    // CHECK: ttnn.add
    // CHECK: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.dequantize"(%1) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112xf32>
    %5 = "ttir.quantize"(%arg1) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %7 = "ttir.dequantize"(%5) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112xf32>
    %9 = "ttir.add"(%3, %7) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %9 : tensor<1x64x112x112xf32>
  }
  func.func @commute_dequantize_past_add_unsuccessful(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    // The quantized types of the operands do not align, so do not commute dequantize down.
    // CHECK-LABEL: @commute_dequantize_past_add_unsuccessful
    // CHECK: ttnn.quantize
    // CHECK: ttnn.dequantize
    // CHECK: ttnn.quantize
    // CHECK: ttnn.dequantize
    // CHECK: ttnn.add
    %1 = "ttir.quantize"(%arg0) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.dequantize"(%1) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112xf32>
    %5 = "ttir.quantize"(%arg1) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.2>>
    %7 = "ttir.dequantize"(%5) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.2>>) -> tensor<1x64x112x112xf32>
    %9 = "ttir.add"(%3, %7) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %9 : tensor<1x64x112x112xf32>
  }
  func.func @commute_dequantize_past_add_one_operand_float(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    // Commute the dequantize down and quantize arg1 with the same scale/zero point as arg0 (see %1).
    // CHECK-LABEL: @commute_dequantize_past_add_one_operand_float
    // CHECK: ttnn.quantize
    // CHECK: ttnn.quantize
    // CHECK: ttnn.add
    // CHECK: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.dequantize"(%1) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112xf32>
    %9 = "ttir.add"(%3, %arg1) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %9 : tensor<1x64x112x112xf32>
  }
  func.func @commute_dequantize_past_maxpool2d_and_merge_qdq(%arg0: tensor<1x112x112x64xf32>) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1>> {
    // It is safe to commute past maxpool2d and merge quantize and dequantize (fold the ops as the scales match).
    // CHECK-LABEL: @commute_dequantize_past_maxpool2d_and_merge_qdq
    // CHECK: ttnn.quantize
    // CHECK: ttnn.max_pool2d
    // CHECK-NOT: ttnn.requantize
    // CHECK-NOT: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0) : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.dequantize"(%1) : (tensor<1x112x112x64x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x112x112x64xf32>
    %5 = "ttir.max_pool2d"(%3) <{kernel = array<i32: 3, 3>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, ceil_mode = false}> : (tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32>
    %7 = "ttir.quantize"(%5) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1>>
    return %7 : tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1>>
  }
  func.func @commute_dequantize_past_maxpool2d_and_merge_qdq_to_requant(%arg0: tensor<1x112x112x64xf32>) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.2>> {
    // It is safe to commute past maxpool2d and merge quantize and dequantize to requantize (scales do not match).
    // CHECK-LABEL: @commute_dequantize_past_maxpool2d_and_merge_qdq_to_requant
    // CHECK: ttnn.quantize
    // CHECK: ttnn.max_pool2d
    // CHECK: ttnn.requantize
    // CHECK-NOT: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0) : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.dequantize"(%1) : (tensor<1x112x112x64x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x112x112x64xf32>
    %5 = "ttir.max_pool2d"(%3) <{kernel = array<i32: 3, 3>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, ceil_mode = false}> : (tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32>
    %7 = "ttir.quantize"(%5) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.2>>
    return %7 : tensor<1x56x56x64x!quant.uniform<i8:f32, 0.2>>
  }
  func.func @commute_dequantize_past_per_tensor_convolution(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> {
    // CHECK-LABEL: func.func @commute_dequantize_past_per_tensor_convolution
    // CHECK: ttnn.quantize
    // CHECK: ttnn.quantize
    // CHECK: ttnn.conv2d
    // CHECK: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0) : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.dequantize"(%1) : (tensor<1x3x224x224x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x3x224x224xf32>
    %5 = "ttir.quantize"(%arg1) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7x!quant.uniform<i8:f32, 0.1>>
    %7 = "ttir.dequantize"(%5) : (tensor<64x3x7x7x!quant.uniform<i8:f32, 0.1>>) -> tensor<64x3x7x7xf32>
    %9 = "ttir.conv2d"(%3, %7) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    return %9 : tensor<1x64x112x112xf32>
  }
  func.func @commute_dequantize_past_per_tensor_convolution_per_axis_weights(%arg0: tensor<1x3x224x224xf32>,%arg1: tensor<3x3x7x7xf32>) -> tensor<1x3x112x112xf32> {
    // CHECK-LABEL: func.func @commute_dequantize_past_per_tensor_convolution_per_axis_weights
    // CHECK: ttnn.quantize
    // CHECK: ttnn.quantize
    // CHECK: ttnn.conv2d
    // CHECK: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0): (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 2.0787402987480164e-02>>
    %3 = "ttir.dequantize"(%1): (tensor<1x3x224x224x!quant.uniform<i8:f32, 2.0787402987480164e-02>>) -> tensor<1x3x224x224xf32>
    %5 = "ttir.quantize"(%arg1) : (tensor<3x3x7x7xf32>) -> tensor<3x3x7x7x!quant.uniform<i8:f32:0, {1.5e-2, 1.0e-2, 8.0e-3}>>
    %7 = "ttir.dequantize"(%5) : (tensor<3x3x7x7x!quant.uniform<i8:f32:0, {1.5e-2, 1.0e-2, 8.0e-3}>>) -> tensor<3x3x7x7xf32>
    %9 = "ttir.conv2d"(%3, %7) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xf32>, tensor<3x3x7x7xf32>) -> tensor<1x3x112x112xf32>
    return %9 : tensor<1x3x112x112xf32>
  }
  func.func @commute_dequantize_past_per_tensor_convolution_per_axis_weights_with_bias(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<3x3x7x7xf32>, %arg2: tensor<1x3x1x1xf32>) -> tensor<1x3x112x112xf32> {
    // CHECK-LABEL: func.func @commute_dequantize_past_per_tensor_convolution_per_axis_weights_with_bias
    // CHECK: ttnn.quantize
    // CHECK: ttnn.quantize
    // CHECK: ttnn.quantize
    // CHECK: ttnn.conv2d
    // CHECK: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0): (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 2.0787402987480164e-02>>
    %3 = "ttir.dequantize"(%1): (tensor<1x3x224x224x!quant.uniform<i8:f32, 2.0787402987480164e-02>>) -> tensor<1x3x224x224xf32>
    %5 = "ttir.quantize"(%arg1) : (tensor<3x3x7x7xf32>) -> tensor<3x3x7x7x!quant.uniform<i8:f32:0, {1.5e-2, 1.0e-2, 8.0e-3}>>
    %7 = "ttir.dequantize"(%5) : (tensor<3x3x7x7x!quant.uniform<i8:f32:0, {1.5e-2, 1.0e-2, 8.0e-3}>>) -> tensor<3x3x7x7xf32>
    %9 = "ttir.conv2d"(%3, %7, %arg2) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xf32>, tensor<3x3x7x7xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x112x112xf32>
    return %9 : tensor<1x3x112x112xf32>
  }
  func.func @commute_dequantize_past_per_tensor_convolution_per_axis_weights_unsuccessful(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> {
    // The number of per-axis weight scales is not the same as the required axis size so a sandwich is applied after commute.
    // CHECK-LABEL: func.func @commute_dequantize_past_per_tensor_convolution_per_axis_weights_unsuccessful
    // CHECK: ttnn.quantize
    // CHECK: ttnn.dequantize
    // CHECK: ttnn.quantize
    // CHECK: ttnn.dequantize
    // CHECK: ttnn.conv2d
    // CHECK: ttnn.quantize
    // CHECK: ttnn.dequantize
    %1 = "ttir.quantize"(%arg0) : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.dequantize"(%1) : (tensor<1x3x224x224x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x3x224x224xf32>
    %5 = "ttir.quantize"(%arg1) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7x!quant.uniform<i8:f32:1, {1.000000e-02:0,2.000000e-02:0,3.000000e-02:0}>>
    %7 = "ttir.dequantize"(%5) : (tensor<64x3x7x7x!quant.uniform<i8:f32:1, {1.000000e-02:0, 2.000000e-02:0, 3.000000e-02:0}>>) -> tensor<64x3x7x7xf32>
    %9 = "ttir.conv2d"(%3, %7) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    return %9 : tensor<1x64x112x112xf32>
  }
}
