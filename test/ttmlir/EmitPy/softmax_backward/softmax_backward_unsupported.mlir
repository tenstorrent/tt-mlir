// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

// CHECK: failed to legalize operation 'ttnn.softmax_backward'
module {
  func.func @softmax_backward(%softmax_output: tensor<1x1x128x256xbf16>, %grad: tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16> {
    %result = "ttcore.composite"(%softmax_output, %grad) <{composite_name = "softmax_backward", decomposition = @softmax_backward_decomp, composite_attributes = {dimension = -1 : si32}}> : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16>
    return %result : tensor<1x1x128x256xbf16>
  }
  func.func private @softmax_backward_decomp(%softmax_output: tensor<1x1x128x256xbf16>, %grad: tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16> {
    return %grad : tensor<1x1x128x256xbf16>
  }
}
