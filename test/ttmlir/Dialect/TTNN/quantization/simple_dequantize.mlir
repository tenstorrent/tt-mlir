// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @dequantize_test(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>) -> tensor<1x3x320x320xf32> {
    // CHECK-LABEL: func.func @dequantize_test(
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    // CHECK: "ttnn.dequantize"(%arg0)
    // CHECK-SAME: {output_dtype = #tt.supportedDataTypes<f32>, scale = 1.000000e-01 : f32, zero_point = 0 : i32}
    // CHECK-SAME: tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>,
    // CHECK-SAME: -> tensor<1x3x320x320xf32,
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    return %1 : tensor<1x3x320x320xf32>
  }
}
