// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s

module @jit_dequantize {
  func.func @dequantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224xf32> {
    // CHECK-LABEL: func.func @dequantize_per_tensor_scale_per_tensor_zp
    %0 = ttir.empty() : tensor<1x3x224x224xf32>
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<1.000000e-01> : tensor<1xf32>
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<0> : tensor<1xsi32>
    // CHECK: "ttir.dequantize_unrolled"
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>, tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    return %1 : tensor<1x3x224x224xf32>
  }
  func.func @dequantize_per_axis_scale_per_axis_zp(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02:10,1.000000e-02:20,5.000000e-03:30}>>) -> tensor<3x3x7x7xf32> {
    // CHECK-LABEL: func.func @dequantize_per_axis_scale_per_axis_zp
    %0 = ttir.empty() : tensor<3x3x7x7xf32>
    // CHECK: "ttir.constant"
    // CHECK-SAME: 2.000000e-02, 1.000000e-02, 5.000000e-03
    // CHECK-SAME: -> tensor<3xf32>
    // CHECK: "ttir.constant"
    // CHECK-SAME: 10, 20, 30
    // CHECK-SAME: -> tensor<3xsi32>
    // CHECK: "ttir.dequantize_unrolled"
    // CHECK-SAME: axis = 1 : i32
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02:10,1.000000e-02:20,5.000000e-03:30}>>, tensor<3x3x7x7xf32>) -> tensor<3x3x7x7xf32>
    return %1 : tensor<3x3x7x7xf32>
  }
}
