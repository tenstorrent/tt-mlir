// RUN: ttmlir-opt --tt-register-device="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x3x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x3x!tt.tile<32x32, bf16>, #dram>, <interleaved>>

#matmul_program_config = #ttnn.matmul_multi_core_reuse_program_config<
  compute_with_storage_grid_size = #ttnn.core_coord<0 : i32, 0 : i32>,
  in0_block_w = 0: i32,
  out_subblock_h = 0 : i32,
  out_subblock_w = 0 : i32,
  per_core_m = 0 : i32,
  per_core_n = 0 : i32
>

module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16, #ttnn_layout>, %arg1: tensor<128x96xbf16, #ttnn_layout1>) -> tensor<64x96xbf16, #ttnn_layout2> {
    %0 = "ttnn.matmul"(%arg0, %arg1)
      <{
        transpose_a = false,
        transpose_b = false,
        matmul_program_config = #matmul_program_config
      }> : (tensor<64x128xbf16, #ttnn_layout>, tensor<128x96xbf16, #ttnn_layout1>) -> tensor<64x96xbf16, #ttnn_layout2>
    return %0 : tensor<64x96xbf16, #ttnn_layout2>
  }
}
