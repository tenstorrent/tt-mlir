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
