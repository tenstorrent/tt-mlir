#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2xf32, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func @constant_row_major() -> tensor<2x2xf32, #ttnn_layout2> {
    %0 = "ttnn.constant"() <{value = dense<"0x0000803F000000400000404000008040"> : tensor<2x2xf32>}> : () -> tensor<2x2xf32, #ttnn_layout1>
    %1 = "ttnn.to_layout"(%0)  : (tensor<2x2xf32, #ttnn_layout1>) -> tensor<2x2xf32, #ttnn_layout2>
    return %1 : tensor<2x2xf32, #ttnn_layout2>
  }
  func.func @constant_tile() -> tensor<2x2xf32, #ttnn_layout2> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.constant"(%0) <{value = dense<"0x0000803F000000400000404000008040"> : tensor<2x2xf32>}> : (!ttnn.device) -> tensor<2x2xf32, #ttnn_layout2>
    return %1 : tensor<2x2xf32, #ttnn_layout2>
  }
}
