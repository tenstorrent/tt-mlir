// RUN: ttmlir-opt -canonicalize %s
module {
  func.func @quantize_identity_per_tensor_scale_per_tensor_zp(%arg0: tensor<2x3x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<2x3x!quant.uniform<i32:f32, 2.000000e-02>> {
    // CHECK-NOT: "ttir.quantize"
    %0 = ttir.empty() : tensor<2x3xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<2x3x!quant.uniform<i32:f32, 2.000000e-02>>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %2 = ttir.empty() : tensor<2x3x!quant.uniform<i32:f32, 2.000000e-02>>
    %3 = "ttir.quantize"(%1, %2) : (tensor<2x3xf32>, tensor<2x3x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<2x3x!quant.uniform<i32:f32, 2.000000e-02>>
    return %3 : tensor<2x3x!quant.uniform<i32:f32, 2.000000e-02>>
  }

  func.func @quantize_identity_per_axis_scale_per_axis_zp(%arg0: tensor<2x3x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>) -> tensor<2x3x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>> {
    // CHECK-NOT: "ttir.quantize"
    %0 = ttir.empty() : tensor<2x3xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<2x3x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %2 = ttir.empty() : tensor<2x3x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
    %3 = "ttir.quantize"(%1, %2) : (tensor<2x3xf32>, tensor<2x3x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>) -> tensor<2x3x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
    return %3 : tensor<2x3x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
  }
}
