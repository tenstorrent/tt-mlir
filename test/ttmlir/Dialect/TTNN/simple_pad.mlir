// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x128x128x384xf32>) -> tensor<1x132x132x384xf32> {
    // CHECK: ttnn.pad
    // CHECK: padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 4 : i32, 4 : i32, 0 : i32], value = 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x132x132x384xf32>
    %1 = "ttir.pad"(%arg0, %0) <{padding = array<i32: 0, 0, 0, 0, 0, 4, 4, 0>, value = 0.000000e+00 : f32}> : (tensor<1x128x128x384xf32>, tensor<1x132x132x384xf32>) -> tensor<1x132x132x384xf32>
    return %1 : tensor<1x132x132x384xf32>
  }
}
