// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2304 + d1 * 256 + d2, d3), <1x1>, memref<504x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

#matmul_program_config = #ttnn.matmul_multi_core_reuse_program_config<
  compute_with_storage_grid_size = #ttnn.core_coord<7, 9>,
  in0_block_w = 8,
  out_subblock_h = 1,
  out_subblock_w = 8,
  per_core_m = 8,
  per_core_n = 8
>

module attributes {} {
  func.func @forward(%arg0: tensor<7x9x256x256xbf16, #ttnn_layout>, %arg1: tensor<7x9x256x256xbf16, #ttnn_layout>) -> tensor<7x9x256x256xbf16, #ttnn_layout> {
    %0 = "ttnn.matmul"(%arg0, %arg1) <{
      transpose_a = false,
      transpose_b = false,
      matmul_program_config = #matmul_program_config}
    > : (tensor<7x9x256x256xbf16, #ttnn_layout>, tensor<7x9x256x256xbf16, #ttnn_layout>) -> tensor<7x9x256x256xbf16, #ttnn_layout>
    return %0 : tensor<7x9x256x256xbf16, #ttnn_layout>
  }
}
