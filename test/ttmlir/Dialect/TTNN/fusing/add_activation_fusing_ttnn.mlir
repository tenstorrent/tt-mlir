// RUN: ttmlir-opt --ttcore-register-device --ttnn-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// relu whose input is an add with no other user is fused into the add as a
// post-activation; an add with a second user is left alone.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  // CHECK-LABEL: func.func @add_relu
  func.func @add_relu(%arg0: tensor<1x1x1024x16xbf16, #ttnn_layout>, %arg1: tensor<1x1x1024x16xbf16, #ttnn_layout>) -> tensor<1x1x1024x16xbf16, #ttnn_layout> {
    // CHECK: %[[ADD:.*]] = "ttnn.add"(%arg0, %arg1)
    // CHECK-SAME: activations = [#ttnn.unary_with_param<op_type = relu>]
    // CHECK-NOT: "ttnn.relu"
    // CHECK: return %[[ADD]]
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<1x1x1024x16xbf16, #ttnn_layout>, tensor<1x1x1024x16xbf16, #ttnn_layout>) -> tensor<1x1x1024x16xbf16, #ttnn_layout>
    %1 = "ttnn.relu"(%0) : (tensor<1x1x1024x16xbf16, #ttnn_layout>) -> tensor<1x1x1024x16xbf16, #ttnn_layout>
    return %1 : tensor<1x1x1024x16xbf16, #ttnn_layout>
  }

  // CHECK-LABEL: func.func @add_two_users
  func.func @add_two_users(%arg0: tensor<1x1x1024x16xbf16, #ttnn_layout>, %arg1: tensor<1x1x1024x16xbf16, #ttnn_layout>) -> (tensor<1x1x1024x16xbf16, #ttnn_layout>, tensor<1x1x1024x16xbf16, #ttnn_layout>) {
    // CHECK: "ttnn.add"(%arg0, %arg1)
    // CHECK-NOT: activations
    // CHECK: "ttnn.relu"
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<1x1x1024x16xbf16, #ttnn_layout>, tensor<1x1x1024x16xbf16, #ttnn_layout>) -> tensor<1x1x1024x16xbf16, #ttnn_layout>
    %1 = "ttnn.relu"(%0) : (tensor<1x1x1024x16xbf16, #ttnn_layout>) -> tensor<1x1x1024x16xbf16, #ttnn_layout>
    return %0, %1 : tensor<1x1x1024x16xbf16, #ttnn_layout>, tensor<1x1x1024x16xbf16, #ttnn_layout>
  }
}
