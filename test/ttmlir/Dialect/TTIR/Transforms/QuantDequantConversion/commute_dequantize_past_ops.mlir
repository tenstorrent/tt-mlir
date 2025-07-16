// RUN: ttmlir-opt --ttir-quant-dequant-conversion %s | FileCheck %s
module {
  func.func @commute_dequantize_past_avgpool2d(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    // It is not safe to commute past avgpool2d, so a avgpool2d -> quantize -> dequantize is performed.
    // CHECK-LABEL: @commute_dequantize_past_avgpool2d
    // CHECK: ttir.empty()
    // CHECK: ttir.quantize
    // CHECK: ttir.dequantize
    // CHECK: ttir.pooling
    // CHECK: ttir.quantize
    // CHECK: ttir.dequantize
    %0 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %2 = ttir.empty() : tensor<1x64x112x112xf32>
    %3 = "ttir.dequantize"(%1, %2) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %4 = ttir.empty() : tensor<1x64x112x112xf32>
    %5 = "ttir.pooling"(%3, %4) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Average>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1>, window_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %5 : tensor<1x64x112x112xf32>
  }
  func.func @commute_dequantize_past_two_maxpool2d(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x28x28xf32> {
    // It is safe to commute past maxpool2d.
    // CHECK-LABEL: @commute_dequantize_past_two_maxpool2d
    %0 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %2 = ttir.empty() : tensor<1x64x112x112xf32>
    %3 = "ttir.dequantize"(%1, %2) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %4 = ttir.empty() : tensor<1x64x56x56xf32>
    %5 = "ttir.pooling"(%3, %4) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> : (tensor<1x64x112x112xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %6 = ttir.empty(): tensor<1x64x28x28xf32>
    %7 = "ttir.pooling"(%5, %6) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> : (tensor<1x64x56x56xf32>, tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
    return %7 : tensor<1x64x28x28xf32>
  }
  func.func @commute_dequantize_past_multi_output_maxpool2d(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<1x64x112x112xf32>) -> (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) {
    // CHECK-LABEL: @commute_dequantize_past_multi_output_maxpool2d
    %0 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %2 = ttir.empty() : tensor<1x64x112x112xf32>
    %3 = "ttir.dequantize"(%1, %2) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %4 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %5 = "ttir.quantize"(%arg1, %4) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %6 = ttir.empty() : tensor<1x64x112x112xf32>
    %7 = "ttir.dequantize"(%5, %6) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %8 = ttir.empty() : tensor<1x64x56x56xf32>
    %9 = ttir.empty() : tensor<1x64x56x56xf32>
    %10, %11 = "ttir.pooling"(%3, %7, %8, %9) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 2, 2>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>, tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>)
    return %10, %11 : tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>
  }
  func.func @commute_dequantize_past_maximum(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    // It is not safe to commute past maximum, so a maximum -> quantize -> dequantize is performed.
    // CHECK-LABEL: @commute_dequantize_past_maximum
    %0 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %2 = ttir.empty() : tensor<1x64x112x112xf32>
    %3 = "ttir.dequantize"(%1, %2) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %4 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x64x112x112xf32>}> : () -> tensor<1x64x112x112xf32>
    %5 = ttir.empty() : tensor<1x64x112x112xf32>
    %6 = "ttir.maximum"(%3, %4, %5) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %6 : tensor<1x64x112x112xf32>
  }
  func.func @commute_dequantize_past_add(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    // It is not safe to commute past add, so a add -> quantize -> dequantize is performed.
    // CHECK-LABEL: @commute_dequantize_past_add
    %0 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %2 = ttir.empty() : tensor<1x64x112x112xf32>
    %3 = "ttir.dequantize"(%1, %2) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %4 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %5 = "ttir.quantize"(%arg1, %4) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %6 = ttir.empty() : tensor<1x64x112x112xf32>
    %7 = "ttir.dequantize"(%5, %6) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %8 = ttir.empty() : tensor<1x64x112x112xf32>
    %9 = "ttir.add"(%3, %7, %8) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %9 : tensor<1x64x112x112xf32>
  }
}
