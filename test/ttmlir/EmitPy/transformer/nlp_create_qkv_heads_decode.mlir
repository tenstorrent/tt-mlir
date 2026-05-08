// RUN: ttmlir-opt --convert-ttnn-to-emitpy -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#dram_interleaved_encoding_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

#l1_height_sharded = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <32x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0, 0), (7, 3)>]>>

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @forward(%input: tensor<1x1x32x3072xbf16, #dram_interleaved_encoding_in>) -> (tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>) {
    %0, %1, %2 = "ttnn.nlp_create_qkv_heads_decode"(%input) <{ num_heads = 32 : ui32, num_kv_heads = 32 : ui32, overlap_qk_coregrid = true }> : (tensor<1x1x32x3072xbf16, #dram_interleaved_encoding_in>) -> (tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>)
    return %0, %1, %2 : tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>
  }
}
