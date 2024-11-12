// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#l1 = #ttnn.buffer_type<l1>
#system = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<8x16xf32, #system>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<8x16xf32, #l1>, <interleaved>>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout1> {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<64x128xf32, #ttnn_layout1>
    // CHECK: %[[C:.*]] = "ttnn.relu"[[C:.*]]
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout1>) -> tensor<64x128xf32, #ttnn_layout1>
    return %1 : tensor<64x128xf32, #ttnn_layout1>
  }
}
