// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s

module @jit_requantize {
  func.func @test_uniform_requantize(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>> {
    %0 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>, tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>
    return %1 : tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>
  }
  func.func @test_per_channel_requantize(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02,5.000000e-03,2.500000e-03}>> {
    %0 = ttir.empty() : tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02,5.000000e-03,2.500000e-03}>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>>, tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02,5.000000e-03,2.500000e-03}>>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02,5.000000e-03,2.500000e-03}>>
    return %1 : tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02,5.000000e-03,2.500000e-03}>>
  }
}
