// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @main(%arg0: tensor<1x1x5x5xbf16>) -> tensor<1x1x7x7xbf16> {
    %0 = ttir.empty() : tensor<1x1x7x7xbf16>
    // CHECK: ttnn.pad
    // CHECK: padding = array<i32: 0, 0, 0, 0, 1, 1, 1, 1>
    %1 = "ttir.pad"(%arg0, %0) <{padding = array<i32: 0, 0, 0, 0, 1, 1, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<1x1x5x5xbf16>, tensor<1x1x7x7xbf16>) -> tensor<1x1x7x7xbf16>
    return %1 : tensor<1x1x7x7xbf16>
  }
}
