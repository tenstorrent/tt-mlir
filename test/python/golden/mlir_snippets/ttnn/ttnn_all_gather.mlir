#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 32 + d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 128 + d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x4>]>} {
  func.func @forward(%arg0: tensor<1x1x32x32xbf16, #ttnn_layout>) -> tensor<1x1x32x128xbf16, #ttnn_layout1> {
    %0 = "ttnn.all_gather"(%arg0) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x32x32xbf16, #ttnn_layout>) -> tensor<1x1x32x128xbf16, #ttnn_layout1>
    return %0 : tensor<1x1x32x128xbf16, #ttnn_layout1>
  }
}
