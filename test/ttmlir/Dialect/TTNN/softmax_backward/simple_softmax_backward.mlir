// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="composite-resolution=force-promote" -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK: "ttnn.softmax_backward"
module {
  func.func @softmax_backward(%softmax_output: tensor<1x1x128x256xbf16>, %grad: tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16> {
    %result = "ttcore.composite"(%softmax_output, %grad) <{composite_name = "softmax_backward", decomposition = @softmax_backward_decomp, composite_attributes = {dimension = -1 : si32}}> : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16>
    return %result : tensor<1x1x128x256xbf16>
  }
  func.func private @softmax_backward_decomp(%softmax_output: tensor<1x1x128x256xbf16>, %grad: tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16> {
    return %grad : tensor<1x1x128x256xbf16>
  }
}
