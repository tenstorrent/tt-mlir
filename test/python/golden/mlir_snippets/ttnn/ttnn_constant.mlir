#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @model() -> tensor<1x2xf32, #ttnn_layout> {
    %0 = "ttnn.constant"() <{ value = dense<[[0.0, 0.0]]> : tensor<1x2xbf16>, dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : () -> tensor<1x2xbf16, #ttnn_layout_dram_rm>
    return %0 : tensor<1x2xbf16, #ttnn_layout_dram_rm>
  }
}
