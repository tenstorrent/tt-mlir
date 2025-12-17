#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @binary_op_fn(%arg0: tensor<128x128xf32, #ttnn_layout>, %arg1: tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout> {
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128x128xf32, #ttnn_layout>, tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout>
    return %0 : tensor<128x128xf32, #ttnn_layout>
  }
}
