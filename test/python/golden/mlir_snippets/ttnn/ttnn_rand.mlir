#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @rand() -> tensor<32x32xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.rand"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, high = 1.000000e+00 : f32, layout = #ttnn.layout<tile>, low = 0.000000e+00 : f32, memory_config = #ttnn.memory_config<#dram, <interleaved>>, seed = 0 : ui32, size = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    return %1 : tensor<32x32xbf16, #ttnn_layout>
  }
}
