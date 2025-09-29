// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#l1 = #ttnn.buffer_type<l1>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x40x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x40x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

#sharded_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x8>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>
#sharded_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x8>, memref<32x5x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>
#sharded_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x8>, memref<1x5x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>

#matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<
  in0_block_w = 1,
  per_core_m = 1,
  per_core_n = 5,
  fused_activation = #ttnn.unary_with_param<op_type = relu>
>

module attributes {} {
  func.func @forward(%arg0: tensor<32x1024xbf16, #ttnn_layout>, %arg1: tensor<1024x1280xbf16, #ttnn_layout1>) -> tensor<32x1280xbf16, #sharded_layout2> {
    %0 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <width_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (7, 0)>]>, <32x128>, <row_major>>>}> : (tensor<32x1024xbf16, #ttnn_layout>) -> tensor<32x1024xbf16, #sharded_layout>
    %1 = "ttnn.to_memory_config"(%arg1) <{memory_config = #ttnn.memory_config<#l1, <width_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (7, 0)>]>, <1024x160>, <row_major>>>}> : (tensor<1024x1280xbf16, #ttnn_layout1>) -> tensor<1024x1280xbf16, #sharded_layout1>
    %2 = "ttnn.matmul"(%0, %1)
      <{
        transpose_a = false,
        transpose_b = false,
        matmul_program_config = #matmul_program_config
      }> : (tensor<32x1024xbf16, #sharded_layout>, tensor<1024x1280xbf16, #sharded_layout1>) -> tensor<32x1280xbf16, #sharded_layout2>
    return %2 : tensor<32x1280xbf16, #sharded_layout2>
  }
}
