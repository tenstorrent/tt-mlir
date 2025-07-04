// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

#matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<
  compute_with_storage_grid_size = #ttnn.core_coord<8, 8>,
  in0_block_w = 16,
  out_subblock_h = 2,
  out_subblock_w = 4,
  out_block_h = 2,
  out_block_w = 4,
  per_core_m = 2,
  per_core_n = 4,
  transpose_mcast = true,
  fused_activation = #ttnn.unary_with_param<op_type = relu>,
  fuse_batch = true
>

module attributes {} {
  func.func @forward(%arg0: tensor<512x512xbf16, #ttnn_layout>, %arg1: tensor<512x1024xbf16, #ttnn_layout1>) -> tensor<512x1024xbf16, #ttnn_layout1> {
    %0 = "ttnn.matmul"(%arg0, %arg1) <{
      transpose_a = false,
      transpose_b = false,
      matmul_program_config = #matmul_program_config
    }> : (tensor<512x512xbf16, #ttnn_layout>, tensor<512x1024xbf16, #ttnn_layout1>) -> tensor<512x1024xbf16, #ttnn_layout1>
    return %0 : tensor<512x1024xbf16, #ttnn_layout1>
  }
}
