// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

module @jit_requantize {
  func.func @test_uniform_requantize(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>> {
    // CHECK-LABEL: func.func @test_uniform_requantize
    %0 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>
    // CHECK: "ttir.constant"
    // CHECK-SAME: 2.000000e-02
    // CHECK-SAME: -> tensor<1xf32>
    // CHECK: "ttir.constant"
    // CHECK-SAME: 0
    // CHECK-SAME: -> tensor<1xi32>
    // CHECK: "ttir.constant"
    // CHECK-SAME: 0.00999999977
    // CHECK-SAME: -> tensor<1xf32>
    // CHECK: "ttir.constant"
    // CHECK-SAME: 0
    // CHECK-SAME: -> tensor<1xi32>
    // CHECK: "ttir.requantize_unrolled"
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>, tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>
    return %1 : tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-02>>
  }
  func.func @test_per_axis_requantize(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02:10,1.000000e-02:20,5.000000e-03:30}>>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02:10,5.000000e-03:20,2.500000e-03:30}>> {
    // CHECK-LABEL: func.func @test_per_axis_requantize
    %0 = ttir.empty() : tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02:10,5.000000e-03:20,2.500000e-03:30}>>
    // CHECK: "ttir.constant"
    // CHECK-SAME: 2.000000e-02, 0.00999999977, 5.000000e-03
    // CHECK-SAME: -> tensor<3xf32>
    // CHECK: "ttir.constant"
    // CHECK-SAME: 10, 20, 30
    // CHECK-SAME: -> tensor<3xi32>
    // CHECK: "ttir.requantize_unrolled"
    // CHECK-SAME: axis = 0 : i32
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02:10,1.000000e-02:20,5.000000e-03:30}>>, tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02:10,5.000000e-03:20,2.500000e-03:30}>>) -> tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02:10,5.000000e-03:20,2.500000e-03:30}>>
    return %1 : tensor<3x3x7x7x!quant.uniform<i32:f32:0, {1.000000e-02:10,5.000000e-03:20,2.500000e-03:30}>>
  }
}
