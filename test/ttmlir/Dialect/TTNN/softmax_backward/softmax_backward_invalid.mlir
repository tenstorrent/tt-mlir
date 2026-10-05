// RUN: not ttmlir-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: 'ttnn.softmax_backward' op grad shape must match softmax_output shape
module {
  func.func @shape(%y: tensor<1x1x32x64xbf16>, %grad: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttnn.softmax_backward"(%y, %grad) <{dimension = -1 : si32}> : (tensor<1x1x32x64xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.softmax_backward' op only bf16 and f32 element types are supported
module {
  func.func @dtype(%y: tensor<1x1x32x64xi32>, %grad: tensor<1x1x32x64xi32>) -> tensor<1x1x32x64xi32> {
    %0 = "ttnn.softmax_backward"(%y, %grad) <{dimension = -1 : si32}> : (tensor<1x1x32x64xi32>, tensor<1x1x32x64xi32>) -> tensor<1x1x32x64xi32>
    return %0 : tensor<1x1x32x64xi32>
  }
}

// -----

// CHECK: error: 'ttnn.softmax_backward' op dimension must select the last dimension
module {
  func.func @dim(%y: tensor<1x1x32x64xf32>, %grad: tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xf32> {
    %0 = "ttnn.softmax_backward"(%y, %grad) <{dimension = 2 : si32}> : (tensor<1x1x32x64xf32>, tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xf32>
    return %0 : tensor<1x1x32x64xf32>
  }
}
