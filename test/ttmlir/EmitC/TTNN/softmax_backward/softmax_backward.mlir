// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

// CHECK: ttml::metal::softmax_backward({{.*}}, {{.*}}, -1)
module {
  func.func @softmax_backward(%softmax_output: tensor<1x1x128x256xbf16>, %grad: tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16> {
    %result = "ttcore.composite"(%softmax_output, %grad) <{composite_name = "softmax_backward", decomposition = @softmax_backward_decomp, composite_attributes = {dimension = -1 : si32}}> : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16>
    return %result : tensor<1x1x128x256xbf16>
  }
  func.func private @softmax_backward_decomp(%softmax_output: tensor<1x1x128x256xbf16>, %grad: tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16> {
    return %grad : tensor<1x1x128x256xbf16>
  }
}
