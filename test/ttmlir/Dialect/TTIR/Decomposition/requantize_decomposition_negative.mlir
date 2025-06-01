// RUN: not ttmlir-opt --split-input-file --ttir-to-ttir-decomposition %s 2>&1 | FileCheck %s

// -----
module {
  func.func @test_axis_out_of_range_requantize(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.2, 0.3, 0.4}>> {
    // CHECK: error: 'ttir.requantize_unrolled' op Axis value 4 is out of the range [0, 4) for the input tensor of rank 4
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.2, 0.3, 0.4}>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>>, tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.2, 0.3, 0.4}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.2, 0.3, 0.4}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.2, 0.3, 0.4}>>
  }
}

// -----
module {
  func.func @test_mismatched_scales_requantize(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.3, 0.4, 0.5}>> {
    // CHECK: error: 'ttir.requantize_unrolled' op Number of scales (2) does not match the size of the quantized axis (3)
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.3, 0.4, 0.5}>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>>, tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.3, 0.4, 0.5}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.3, 0.4, 0.5}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.3, 0.4, 0.5}>>
  }
}

// -----
module {
  func.func @test_zero_point_out_of_range_requantize(%arg0: tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>>) -> tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.3:0, 0.4:0, 0.5:0}>> {
    // CHECK: error: 'ttir.requantize_unrolled' op Zero point 300 is out of the range for storage type 'i8'
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.3:0, 0.4:0, 0.5:0}>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>>, tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.3:0, 0.4:0, 0.5:0}>>) -> tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.3:0, 0.4:0, 0.5:0}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.3:0, 0.4:0, 0.5:0}>>
  }
}

// -----
module {
  func.func @test_input_output_quantization_types_different_requantize(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2, 0.3}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>> {
    // CHECK: error: 'ttir.requantize_unrolled' op Input and output element types must both be per-axis or both be per-tensor quantized types, but got '!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>' and '!quant.uniform<i32:f32, 2.000000e-01>'
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2, 0.3}>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
  }
}
