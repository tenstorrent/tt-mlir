// RUN: ttmlir-opt --convert-ttnn-to-emitpy -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir

#dram = #ttnn.buffer_type<dram>
#dram_interleaved = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x120x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#dram_interleaved_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 320 + d1 * 32 + d2, d3), <1x1>, memref<10x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @forward(%input: tensor<1x1x32x3840xbf16, #dram_interleaved>) -> (tensor<1x10x32x128xbf16, #dram_interleaved_out>, tensor<1x10x32x128xbf16, #dram_interleaved_out>, tensor<1x10x32x128xbf16, #dram_interleaved_out>) {
    %0, %1, %2 = "ttnn.nlp_create_qkv_heads"(%input) <{num_q_heads = 10 : ui32, num_kv_heads = 10 : ui32}> : (tensor<1x1x32x3840xbf16, #dram_interleaved>) -> (tensor<1x10x32x128xbf16, #dram_interleaved_out>, tensor<1x10x32x128xbf16, #dram_interleaved_out>, tensor<1x10x32x128xbf16, #dram_interleaved_out>)
    return %0, %1, %2 : tensor<1x10x32x128xbf16, #dram_interleaved_out>, tensor<1x10x32x128xbf16, #dram_interleaved_out>, tensor<1x10x32x128xbf16, #dram_interleaved_out>
  }
}
