// RUN: ttmlir-opt --ttir-quant-dequant-conversion %s | FileCheck %s
module {
  func.func @commute_quantize_past_maxpool2d(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>> {
    // CHECK-LABEL: func.func @commute_quantize_past_maxpool2d
    // CHECK: %[[QEMPTY:.*]] = ttir.empty() : tensor<1x64x112x112x!quant.uniform<i8:f32,
    // CHECK: %[[Q:.*]] = "ttir.quantize"(%arg0, %[[QEMPTY]])
    // CHECK: %[[PEMPTY:.*]] = ttir.empty() : tensor<1x64x56x56x!quant.uniform<i8:f32,
    // CHECK: %[[POOL:.*]] = "ttir.pooling"(%[[Q]], %[[PEMPTY]])
    // CHECK: return %[[POOL]]
    %0 = ttir.empty() : tensor<1x64x56x56xf32>
    %1 = "ttir.pooling"(%arg0, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> : (tensor<1x64x112x112xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %2 = ttir.empty() : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.quantize"(%1, %2) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
    return %3 : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
  }
  func.func @commute_quantize_past_relu(%arg0: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>> {
    // CHECK-LABEL: func.func @commute_quantize_past_relu
    // CHECK: %[[ZERO:.+]] = "ttir.constant"()
    // CHECK: %[[QEMPTY1:.+]] = ttir.empty() : tensor<1x64x56x56x!quant.uniform<i8:f32,
    // CHECK: %[[Q1:.+]] = "ttir.quantize"(%arg0, %[[QEMPTY1]])
    // CHECK: %[[QEMPTY2:.+]] = ttir.empty() : tensor<1x64x56x56x!quant.uniform<i8:f32,
    // CHECK: %[[Q2:.+]] = "ttir.quantize"(%[[ZERO]], %[[QEMPTY2]])
    // CHECK: %[[DQEMPTY1:.+]] = ttir.empty() : tensor<1x64x56x56xf32>
    // CHECK: %[[DQ1:.+]] = "ttir.dequantize"(%[[Q1]], %[[DQEMPTY1]])
    // CHECK: %[[DQEMPTY2:.+]] = ttir.empty() : tensor<1x64x56x56xf32>
    // CHECK: %[[DQ2:.+]] = "ttir.dequantize"(%[[Q2]], %[[DQEMPTY2]])
    // CHECK: %[[MAX:.+]] = "ttir.maximum"(%[[DQ1]], %[[DQ2]],
    // CHECK: %[[QEMPTY3:.+]] = ttir.empty() : tensor<1x64x56x56x!quant.uniform<i8:f32,
    // CHECK: %[[Q3:.+]] = "ttir.quantize"(%[[MAX]], %[[QEMPTY3]]) {ttir.skip_qdq_commute}
    // CHECK: return %[[Q3]]
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x64x56x56xf32>}> : () -> tensor<1x64x56x56xf32>
    %1 = ttir.empty() : tensor<1x64x56x56xf32>
    %2 = "ttir.maximum"(%arg0, %0, %1) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %3 = ttir.empty() : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
    %4 = "ttir.quantize"(%2, %3) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
    return %4 : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
  }
  func.func @commute_quantize_past_add(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>> {
    // CHECK-LABEL: func.func @commute_quantize_past_add
    // CHECK: "ttir.quantize"(%arg0
    // CHECK: "ttir.quantize"(%arg1
    // CHECK: "ttir.dequantize"
    // CHECK: "ttir.dequantize"
    // CHECK: "ttir.add"
    // CHECK: "ttir.quantize"
    // CHECK: return
    %0 = ttir.empty() : tensor<1x64x56x56xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %2 = ttir.empty() : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
    %3 = "ttir.quantize"(%1, %2) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
    return %3 : tensor<1x64x56x56x!quant.uniform<i8:f32, 0.1>>
  }
  func.func @commute_dequantize_past_maxpool2d(%arg0: tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x64x56x56xf32> {
    %0 = ttir.empty() : tensor<1x64x112x112xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.1>>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %2 = ttir.empty() : tensor<1x64x56x56xf32>
    %3 = "ttir.pooling"(%1, %2) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> : (tensor<1x64x112x112xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    return %3 : tensor<1x64x56x56xf32>
  }
}
