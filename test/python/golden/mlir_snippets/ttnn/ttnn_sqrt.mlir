#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @model(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %0 = "ttnn.sqrt"(%arg0) : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout>
    return %0 : tensor<32x32xf32, #ttnn_layout>
  }
}
