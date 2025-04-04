// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @quantize_test(%arg0: tensor<1x3x320x320xf32>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>> {
    // CHECK-LABEL: func.func @quantize_test(
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
    // CHECK: "ttnn.quantize"(%arg0)
    // CHECK-SAME: {output_dtype = #tt.supportedDataTypes<si32>, scale = 1.000000e-01 : f32, zero_point = 0 : i32}
    // CHECK-SAME: tensor<1x3x320x320xf32
    // CHECK-SAME: -> tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>,
    %1 = "ttir.quantize"(%arg0, %0) : (tensor<1x3x320x320xf32>, tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>
  }
}
