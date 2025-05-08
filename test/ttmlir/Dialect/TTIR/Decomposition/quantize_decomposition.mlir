// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s

module attributes {} {
  func.func @quantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>> {
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
  }
  func.func @quantize_per_axis_scale_per_axis_zp(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>> {
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32:1, {1.000000e-01:10,2.000000e-01:20,3.000000e-01:30}>>
  }
}
