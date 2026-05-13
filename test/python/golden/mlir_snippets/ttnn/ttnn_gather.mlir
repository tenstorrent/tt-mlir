#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_index = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
module {
  func.func @model(%arg0: tensor<32x256xf32, #ttnn_layout>, %arg1: tensor<32x32xui32, #ttnn_layout_index>) -> tensor<32x32xf32, #ttnn_layout_index> {
    %0 = "ttnn.gather"(%arg0, %arg1) <{dim = 1 : i32}> : (tensor<32x256xf32, #ttnn_layout>, tensor<32x32xui32, #ttnn_layout_index>) -> tensor<32x32xf32, #ttnn_layout_index>
    return %0 : tensor<32x32xf32, #ttnn_layout_index>
  }
}
