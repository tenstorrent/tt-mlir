// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s

module @jit_dequantize {
  func.func @dequantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224xf32> {
    %0 = ttir.empty() : tensor<1x3x224x224xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>, tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    return %1 : tensor<1x3x224x224xf32>
  }
  func.func @dequantize_per_axis_scale_per_tensor_zp(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>>) -> tensor<3x3x7x7xf32> {
    %0 = ttir.empty() : tensor<3x3x7x7xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>>, tensor<3x3x7x7xf32>) -> tensor<3x3x7x7xf32>
    return %1 : tensor<3x3x7x7xf32>
  }
  func.func @dequantize_per_axis_scale_per_channel_zp(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02:10,1.000000e-02:20,5.000000e-03:30}>>) -> tensor<3x3x7x7xf32> {
    %0 = ttir.empty() : tensor<3x3x7x7xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02:10,1.000000e-02:20,5.000000e-03:30}>>, tensor<3x3x7x7xf32>) -> tensor<3x3x7x7xf32>
    return %1 : tensor<3x3x7x7xf32>
  }
}
