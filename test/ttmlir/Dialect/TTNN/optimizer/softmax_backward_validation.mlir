// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @softmax_backward(%y: tensor<1x1x128x256xbf16>, %grad: tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK-DAG: #[[LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}!ttcore.tile<32x32, bf16>
    // CHECK-LABEL: func.func @softmax_backward(
    // CHECK: %[[R:.*]] = "ttnn.softmax_backward"({{.*}}) <{dimension = -1 : si32}>
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[LAYOUT]]>
    // CHECK: return %[[R]]
    %0 = "ttcore.composite"(%y, %grad) <{composite_name = "softmax_backward", decomposition = @decomp, composite_attributes = {dimension = -1 : si32}}> : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }
  func.func private @decomp(%y: tensor<1x1x128x256xbf16>, %grad: tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xbf16> { return %grad : tensor<1x1x128x256xbf16> }
}
