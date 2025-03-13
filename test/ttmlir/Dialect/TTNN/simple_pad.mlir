// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x1x5x5xbf16>) -> tensor<1x1x7x7xbf16> {
    // CHECK: ttnn.pad
    // CHECK: padding = array<i32: 0, 0, 0, 0, 1, 1, 1, 1>
     %0 = tensor.empty() : tensor<1x1x7x7xbf16>
    %1 = "ttir.pad"(%arg0,%0) <{padding = array<i32: 0, 0, 0, 0, 1, 1, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<1x1x5x5xbf16>,tensor<1x1x7x7xbf16>) -> tensor<1x1x7x7xbf16>
    return %1 : tensor<1x1x7x7xbf16>
  }
}


// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x1x5x5xbf16>) -> tensor<1x1x7x5xbf16> {
    // CHECK: ttnn.pad
    // CHECK: padding = array<i32: 0, 0, 0, 0, 1, 1, 0, 0>
     %0 = tensor.empty() : tensor<1x1x7x5xbf16>
    %1 = "ttir.pad"(%arg0,%0) <{padding = array<i32: 0, 0, 0, 0, 1, 1, 0, 0>, value = 0.000000e+00 : f32}> : (tensor<1x1x5x5xbf16>,tensor<1x1x7x5xbf16>) -> tensor<1x1x7x5xbf16>
    return %1 : tensor<1x1x7x5xbf16>
  }
}




// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x1x5x5xbf16>) -> tensor<1x1x6x6xbf16> {
    // CHECK: ttnn.pad
    // CHECK: padding = array<i32: 0, 0, 0, 0, 0, 1, 0, 1>
     %0 = tensor.empty() : tensor<1x1x6x6xbf16>
    %1 = "ttir.pad"(%arg0,%0) <{padding = array<i32: 0, 0, 0, 0, 0, 1, 0, 1>, value = 0.000000e+00 : f32}> : (tensor<1x1x5x5xbf16>,tensor<1x1x6x6xbf16>) -> tensor<1x1x6x6xbf16>
    return %1 : tensor<1x1x6x6xbf16>
  }
}