// RUN: not ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s 2>&1 | FileCheck %s

module {
  func.func @test_rank_mismatch(%arg0: tensor<3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x320x320xf32> {
    // CHECK: error: 'ttir.dequantize' op Input tensor rank of 3 does not match the output tensor rank of 4
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}

// -----
module {
  func.func @test_shape_mismatch(%arg0: tensor<1x4x320x320x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x320x320xf32> {
    // CHECK: error: 'ttir.dequantize' op Output tensor shape (1,3,320,320) must match the inferred shape: (1,4,320,320)
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x4x320x320x!quant.uniform<i32:f32, 1.000000e-01>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}

// -----
module {
  func.func @test_invalid_input_data_type(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32> {
    // CHECK: error: 'ttir.dequantize' op Input element type must be UniformQuantizedType or UniformQuantizedPerAxisType, but got 'f32'
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}

// -----
module {
  func.func @test_invalid_output_data_type(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x320x320xi32> {
    // CHECK: error: 'ttir.dequantize' op Output element type must be float, but got 'i32'
    %0 = ttir.empty() : tensor<1x3x320x320xi32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>, tensor<1x3x320x320xi32>) -> tensor<1x3x320x320xi32>
    return %1 : tensor<1x3x320x320xi32>
  }
}

// -----
module {
  func.func @test_axis_out_of_range_dequantize(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>>) -> tensor<1x3x320x320xf32> {
    // CHECK: error: 'ttnn.dequantize' op Axis value 4 is out of the range for the input tensor of rank 4
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}

// -----
module {
  func.func @test_mismatched_scales_dequantize(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>>) -> tensor<1x3x320x320xf32> {
    // CHECK: error: 'ttnn.dequantize' op Number of scales (2) does not match the size of the quantized axis (3)
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}

// -----
module {
  func.func @test_zero_point_out_of_range_dequantize(%arg0: tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>>) -> tensor<1x3x320x320xf32> {
    // CHECK: 'ttnn.dequantize' op Zero point 300 is out of the range for storage type 'i8'
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}
