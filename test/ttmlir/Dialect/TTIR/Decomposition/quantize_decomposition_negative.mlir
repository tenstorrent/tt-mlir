// RUN: not ttmlir-opt --split-input-file --ttir-to-ttir-decomposition %s 2>&1 | FileCheck %s

// -----
module {
  func.func @test_axis_out_of_range(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>> {
    // CHECK: error: 'ttir.quantize_unrolled' op Axis value 4 is out of the range [0, 4) for the input tensor of rank 4
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:4, {0.1, 0.2, 0.3}>>
  }
}

// -----
module {
  func.func @test_mismatched_scales(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>> {
    // CHECK: error: 'ttir.quantize_unrolled' op Number of scales (2) does not match the size of the quantized axis (3)
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {0.1, 0.2}>>
  }
}

// -----
module {
  func.func @test_zero_point_out_of_range(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>> {
    // CHECK: error: 'ttir.quantize_unrolled' op Zero point 300 is out of the range for storage type 'i8'
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>>) -> tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i8:f32:1, {0.1:0, 0.2:0, 0.3:300}>>
  }
}
