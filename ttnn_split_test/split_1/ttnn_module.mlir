#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 1024 + d1 * 1024 + d2 * 32 + d3, d4), <1x1>, memref<1024x32xf32, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 1024 + d1 * 1024 + d2 * 32 + d3, d4), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #system_memory>>
module {
  func.func @to_layout_module(%arg0: tensor<1x1x32x32x32xf32, #ttnn_layout> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x32x32x32xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> tensor<1x1x32x32x32xf32, #ttnn_layout1> {
    %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<1x1x32x32x32xf32, #ttnn_layout>) -> tensor<1x1x32x32x32xf32, #ttnn_layout1>
    return %0 : tensor<1x1x32x32x32xf32, #ttnn_layout1>
  }
}
