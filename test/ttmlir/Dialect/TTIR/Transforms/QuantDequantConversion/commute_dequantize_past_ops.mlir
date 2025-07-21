// RUN: ttmlir-opt --ttir-quant-dequant-conversion %s | FileCheck %s
module {
  func.func @commute_dequantize_past_avgpool2d(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    // It is not safe to commute past avgpool2d, so a avgpool2d -> quantize -> dequantize is performed.
    // CHECK-LABEL: @commute_dequantize_past_avgpool2d
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
    // CHECK: ttir.quantize
    // CHECK: ttir.pooling
    // CHECK: ttir.pooling
    // CHECK: ttir.dequantize
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
    // CHECK: ttir.quantize
    // CHECK: ttir.quantize
    // CHECK: ttir.pooling
    // CHECK: ttir.dequantize
    // CHECK: ttir.dequantize
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
    // It is not safe to commute past maximum, so commute dequantize down with q -> dq sandwich.
    // CHECK-LABEL: @commute_dequantize_past_maximum
    // CHECK: ttir.quantize
    // CHECK: ttir.dequantize
    // CHECK: ttir.maximum
    // CHECK: ttir.quantize
    // CHECK: ttir.dequantize
    %0 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %2 = ttir.empty() : tensor<1x64x112x112xf32>
    %3 = "ttir.dequantize"(%1, %2) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %4 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x64x112x112xf32>}> : () -> tensor<1x64x112x112xf32>
    %5 = ttir.empty() : tensor<1x64x112x112xf32>
    %6 = "ttir.maximum"(%3, %4, %5) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %6 : tensor<1x64x112x112xf32>
  }
  func.func @commute_dequantize_past_add_successful(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    // The quantized types of the operands both align, so simply commute dequantize down.
    // CHECK-LABEL: @commute_dequantize_past_add_successful
    // CHECK: ttir.quantize
    // CHECK: ttir.quantize
    // CHECK: ttir.add
    // CHECK: ttir.dequantize
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
  func.func @commute_dequantize_past_add_unsuccessful(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    // The quantized types of the operands do not align, so do not commute dequantize down.
    // CHECK-LABEL: @commute_dequantize_past_add_unsuccessful
    // CHECK: ttir.quantize
    // CHECK: ttir.dequantize
    // CHECK: ttir.quantize
    // CHECK: ttir.dequantize
    // CHECK: "ttir.add"{{.*}} -> tensor<1x64x112x112xf32>
    %0 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %2 = ttir.empty() : tensor<1x64x112x112xf32>
    %3 = "ttir.dequantize"(%1, %2) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %4 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.2>>
    %5 = "ttir.quantize"(%arg1, %4) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.2>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.2>>
    %6 = ttir.empty() : tensor<1x64x112x112xf32>
    %7 = "ttir.dequantize"(%5, %6) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.2>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %8 = ttir.empty() : tensor<1x64x112x112xf32>
    %9 = "ttir.add"(%3, %7, %8) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %9 : tensor<1x64x112x112xf32>
  }
  func.func @commute_dequantize_past_maxpool2d_and_merge_qdq(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>> {
    // It is safe to commute past maxpool2d and merge quantize and dequantize (no op).
    // CHECK-LABEL: @commute_dequantize_past_maxpool2d_and_merge_qdq
    // CHECK: ttir.quantize
    // CHECK: ttir.pooling
    // CHECK-NOT: ttir.dequantize
    %0 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %2 = ttir.empty() : tensor<1x64x112x112xf32>
    %3 = "ttir.dequantize"(%1, %2) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %4 = ttir.empty() : tensor<1x64x56x56xf32>
    %5 = "ttir.pooling"(%3, %4) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> : (tensor<1x64x112x112xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %6 = ttir.empty() : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
    %7 = "ttir.quantize"(%5, %6) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
    return %7 : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
  }
  func.func @commute_dequantize_past_maxpool2d_and_merge_qdq_to_requant(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.2>> {
    // It is safe to commute past maxpool2d and merge quantize and dequantize to requantize.
    // CHECK-LABEL: @commute_dequantize_past_maxpool2d_and_merge_qdq_to_requant
    // CHECK: ttir.quantize
    // CHECK: ttir.pooling
    // CHECK: ttir.requantize
    // CHECK-NOT: ttir.dequantize
    %0 = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>
    %2 = ttir.empty() : tensor<1x64x112x112xf32>
    %3 = "ttir.dequantize"(%1, %2) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %4 = ttir.empty() : tensor<1x64x56x56xf32>
    %5 = "ttir.pooling"(%3, %4) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> : (tensor<1x64x112x112xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %6 = ttir.empty() : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.2>>
    %7 = "ttir.quantize"(%5, %6) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56x!quant.uniform<i8:f32, 0.2>>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.2>>
    return %7 : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.2>>
  }
}
