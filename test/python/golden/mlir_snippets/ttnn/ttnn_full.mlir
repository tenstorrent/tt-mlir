#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func @full_row_major() -> tensor<32x32xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 1.000000e+00 : f32, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout1>) -> tensor<32x32xbf16, #ttnn_layout>
    return %2 : tensor<32x32xbf16, #ttnn_layout>
  }
  func.func @full_tile() -> tensor<32x32xf32, #ttnn_layout2> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, fill_value = 1.000000e+00 : f32, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout2>
    return %1 : tensor<32x32xf32, #ttnn_layout2>
  }
}
