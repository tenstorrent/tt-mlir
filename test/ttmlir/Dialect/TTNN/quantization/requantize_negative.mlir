// RUN: not ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s 2>&1 | FileCheck %s

module {
  func.func @test_rank_mismatch(%arg0: tensor<3x320x320x!quant.uniform<i32:f32, 0.1>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>> {
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    // CHECK: error: 'ttir.requantize' op Input tensor rank of 3 does not match the output tensor rank of 4
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<3x320x320x!quant.uniform<i32:f32, 0.1>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
  }
}

// -----

module {
  func.func @test_shape_mismatch(%arg0: tensor<1x4x320x320x!quant.uniform<i32:f32, 0.1>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>> {
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    // CHECK: error: 'ttir.requantize' op Output tensor shape (1,3,320,320) must match the inferred shape: (1,4,320,320)
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x4x320x320x!quant.uniform<i32:f32, 0.1>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
  }
}

// -----
module {
  func.func @test_invalid_input_type(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>> {
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    // CHECK: error: 'ttir.requantize' op Input element type must be UniformQuantizedType or UniformQuantizedPerAxisType, but got 'f32'
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
  }
}

// -----
module {
  func.func @test_invalid_output_type(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>) -> tensor<1x3x320x320xf32> {
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    // CHECK: error: 'ttir.requantize' op Output element type must be UniformQuantizedType or UniformQuantizedPerAxisType, but got 'f32'
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}

// -----
module {
  func.func @test_axis_out_of_range_requantize(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.2, 0.3, 0.4}>> {
    // CHECK: error: 'ttnn.requantize' op Axis value 4 is out of the range for the input tensor of rank 4
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.2, 0.3, 0.4}>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>>, tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.2, 0.3, 0.4}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.2, 0.3, 0.4}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.2, 0.3, 0.4}>>
  }
}

// -----
module {
  func.func @test_mismatched_scales_requantize(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.3, 0.4, 0.5}>> {
    // CHECK: error: 'ttnn.requantize' op Number of scales (2) does not match the size of the quantized axis (3)
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.3, 0.4, 0.5}>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>>, tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.3, 0.4, 0.5}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.3, 0.4, 0.5}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.3, 0.4, 0.5}>>
  }
}

// -----
module {
  func.func @test_zero_point_out_of_range_requantize(%arg0: tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>>) -> tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.3:0, 0.4:0, 0.5:0}>> {
    // CHECK: error: 'ttnn.requantize' op Zero point 300 is out of the range for storage type 'i8'
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.3:0, 0.4:0, 0.5:0}>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>>, tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.3:0, 0.4:0, 0.5:0}>>) -> tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.3:0, 0.4:0, 0.5:0}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.3:0, 0.4:0, 0.5:0}>>
  }
}

// -----
module {
  func.func @test_input_output_quantization_types_different_requantize(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2, 0.3}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>> {
    // CHECK: error: 'ttnn.requantize' op Input and result element types must both be per-axis or both be per-tensor quantized types, but got '!quant.uniform<i32:f32:1, {1.000000e-01,2.000000e-01,3.000000e-01}>' and '!quant.uniform<i32:f32, 2.000000e-01>'
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2, 0.3}>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
  }
}
