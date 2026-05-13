#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @rms_norm_pre_all_gather_forward(%input: tensor<1x1x128x64xbf16, #ttnn_layout>) -> tensor<1x1x128x32xbf16, #ttnn_layout_out> {
    %0 = "ttnn.rms_norm_pre_all_gather"(%input) <{dtype = #ttcore.supportedDataTypes<bf16>, use_2d_core_grid = false}> : (tensor<1x1x128x64xbf16, #ttnn_layout>) -> tensor<1x1x128x32xbf16, #ttnn_layout_out>
    return %0 : tensor<1x1x128x32xbf16, #ttnn_layout_out>
  }
}
