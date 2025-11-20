#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
module {
  func.func @test(%arg0: tensor<32x32xf32, #ttnn_layout>, %arg1: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %0 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
    %1 = "ttnn.to_memory_config"(%arg1) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>>>}> : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
    %2 = "ttnn.add"(%0, %1) <{dtype = #ttcore.supportedDataTypes<f32>}> {ttnn.hoist_generic_via_d2m} : (tensor<32x32xf32, #ttnn_layout1>, tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout1>
    %3 = "ttnn.to_memory_config"(%2) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout1>) -> tensor<32x32xf32, #ttnn_layout>
    return %3 : tensor<32x32xf32, #ttnn_layout>
  }
}
