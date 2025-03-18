// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @forward(%arg0: tensor<1x3x320x320x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x3x320x320xf32> {
    %0 = tensor.empty() : tensor<1x3x320x320xf32>
    %1 = "ttir.dequantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i8:f32, 0.1>>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    // CHECK: "ttnn.dequantize"
    return %1 : tensor<1x3x320x320xf32>
  }
}
