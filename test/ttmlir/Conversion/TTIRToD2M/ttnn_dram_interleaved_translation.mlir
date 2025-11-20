#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 256 + d2, d3), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @test_lower_interleaved_dram(%arg0: tensor<1024x1024xbf16, #ttnn_layout>, %arg1: tensor<1024x1024xbf16, #ttnn_layout>) -> tensor<1024x1024xbf16, #ttnn_layout> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<1024x1024xbf16, #ttnn_layout>, tensor<1024x1024xbf16, #ttnn_layout>) -> tensor<1024x1024xbf16, #ttnn_layout>
    return %0 : tensor<1024x1024xbf16, #ttnn_layout>
  }
  func.func @test_lower_interleaved_dram_1(%arg0: tensor<2x512x1024xbf16, #ttnn_layout1>, %arg1: tensor<2x512x1024xbf16, #ttnn_layout1>) -> tensor<2x512x1024xbf16, #ttnn_layout1> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<2x512x1024xbf16, #ttnn_layout1>, tensor<2x512x1024xbf16, #ttnn_layout1>) -> tensor<2x512x1024xbf16, #ttnn_layout1>
    return %0 : tensor<2x512x1024xbf16, #ttnn_layout1>
  }
  func.func @test_lower_interleaved_dram_2(%arg0: tensor<2x2x256x1024xbf16, #ttnn_layout2>, %arg1: tensor<2x2x256x1024xbf16, #ttnn_layout2>) -> tensor<2x2x256x1024xbf16, #ttnn_layout2> {
    %0 = "ttir.abs"(%arg0, %arg1) : (tensor<2x2x256x1024xbf16, #ttnn_layout2>, tensor<2x2x256x1024xbf16, #ttnn_layout2>) -> tensor<2x2x256x1024xbf16, #ttnn_layout2>
    return %0 : tensor<2x2x256x1024xbf16, #ttnn_layout2>
  }
}
