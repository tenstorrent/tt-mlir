// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x128x128x384xf32>) -> tensor<1x132x132x384xf32> {
    // CHECK: ttnn.pad
    // CHECK-SAME: padding = array<i32: 0, 0, 2, 2, 2, 2, 0, 0>, value = 0.000000e+00 : f32
    %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 2, 2, 2, 2, 0, 0>, value = 0.000000e+00 : f32}> : (tensor<1x128x128x384xf32>) -> tensor<1x132x132x384xf32>
    return %1 : tensor<1x132x132x384xf32>
  }
}
