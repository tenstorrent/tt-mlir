//proposed empty + toLayout addition for a memory_config = ttnn.DRAM_MEMORY_CONFIG
//is generated after frontend IR gen in ir_generator.py
#l1 = #ttnn.buffer_type<l1>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<4x4x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>
#ttnn_layout_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>, exactGrid = true>
module {
  func.func @abs(%arg0: tensor<1024x1024xbf16, #ttnn_layout>) -> tensor<1024x1024xbf16, #ttnn_layout_dram> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.abs"(%arg0) {ttnn.hoist_generic_via_d2m} : (tensor<1024x1024xbf16, #ttnn_layout>) -> tensor<1024x1024xbf16, #ttnn_layout>
    %2 = "ttir.empty"() : () -> tensor<1024x1024xbf16, #ttnn_layout_dram>
    %3 = "ttir.to_layout"(%1, %2) : (tensor<1024x1024xbf16, #ttnn_layout>, tensor<1024x1024xbf16, #ttnn_layout_dram>) -> tensor<1024x1024xbf16, #ttnn_layout_dram>
    return %3 : tensor<1024x1024xbf16, #ttnn_layout_dram>
  }
}
