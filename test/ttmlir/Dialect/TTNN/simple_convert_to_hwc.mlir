// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @convert_to_hwc(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x1x8x3xf32> {
    // CHECK: "ttnn.convert_to_hwc"(%arg0)
    // CHECK-SAME: tensor<1x2x3x4xf32,
    // CHECK-SAME: -> tensor<1x1x8x3xf32,
    %1 = "ttir.convert_to_hwc"(%arg0) : (tensor<1x2x3x4xf32>) -> tensor<1x1x8x3xf32>
    return %1 : tensor<1x1x8x3xf32>
  }
}
