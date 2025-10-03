// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

#matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<
  compute_with_storage_grid_size = #ttnn.core_coord<8, 2>,
  in0_block_w = 32,
  out_subblock_h = 1,
  out_subblock_w = 2,
  out_block_h = 8,
  out_block_w = 6,
  per_core_m = 8,
  per_core_n = 6,
  fuse_batch = true,
  fused_activation = #ttnn.unary_with_param<op_type = sub_unary_sfpu, params = [1.0 : f32]>,
  mcast_in0 = true,
  gather_in0 = false,
  hop_cores = #ttnn.core_range_set<>,
  num_global_cb_receivers = 0,
  untilize_out = false
>

module attributes {} {
  func.func @forward(%arg0: tensor<256x1024xbf16, #ttnn_layout>, %arg1: tensor<1024x2048xbf16, #ttnn_layout1>) -> tensor<256x2048xbf16, #ttnn_layout2> {
    %0 = "ttnn.matmul"(%arg0, %arg1)
      <{
        transpose_a = false,
        transpose_b = false,
        matmul_program_config = #matmul_program_config
      }> : (tensor<256x1024xbf16, #ttnn_layout>, tensor<1024x2048xbf16, #ttnn_layout1>) -> tensor<256x2048xbf16, #ttnn_layout2>
    return %0 : tensor<256x2048xbf16, #ttnn_layout2>
  }
}
