#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#l1_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
module {
  func.func @model(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout> {
    %0 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>
    return %0 : tensor<32x32xf32, #l1_layout>
  }
}
