// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: ttmlir-opt --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/out.json ttnn-perf-metrics-verbose-output-enabled=true" %s -o /dev/null
// RUN: cat %t.dir/out.json | FileCheck %s

// Per-op FLOP accounting for a single SDPA prefill (no causal mask).
//   Q = [B=1, Hq=8, Sq=512, Dk=128], K = [.., Sk=512, ..], V = [.., Dv=128]
//   flops = 2 * B * Hq * Sq * Sk * (Dk + Dv)
//         = 2 * 1 * 8 * 512 * 512 * (128 + 128) = 1,073,741,824

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{
  role = host, target_triple = "x86_64-pc-linux"
}], [{
  arch = <wormhole_b0>,
  grid = 8x8,
  coord_translation_offsets = 18x18,
  l1_size = 1499136,
  num_dram_channels = 12,
  dram_channel_size = 1073741824,
  noc_l1_address_align_bytes = 16,
  pcie_address_align_bytes = 32,
  noc_dram_address_align_bytes = 32,
  l1_unreserved_base = 103712,
  erisc_l1_unreserved_base = 98304,
  dram_unreserved_base = 1920032,
  dram_unreserved_end = 1073119552,
  supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>],
  supported_tile_sizes = [ 4x16, 16x16, 32x16, 4x32, 16x32, 32x32 ],
  dst_physical_size_tiles = 16,
  num_cbs = 64,
  num_compute_threads = 1,
  num_datamovement_threads = 2,
  dram_grid = 1x12,
  dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)],
  dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]
}], [0], [1 : i32], [ 0x0x0x0]>

#q_full  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 512 + d2, d3), <1x1>, memref<128x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv_full = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 512 + d2, d3), <1x1>, memref<128x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {ttcore.system_desc = #system_desc} {
  func.func @forward(
      %q: tensor<1x8x512x128xbf16, #q_full>  {ttcore.argument_type = #ttcore.argument_type<input>},
      %k: tensor<1x8x512x128xbf16, #kv_full> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.kv_cache},
      %v: tensor<1x8x512x128xbf16, #kv_full> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.kv_cache}
  ) -> tensor<1x8x512x128xbf16, #q_full>
      attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.scaled_dot_product_attention"(%q, %k, %v)
        <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x8x512x128xbf16, #q_full>,
           tensor<1x8x512x128xbf16, #kv_full>,
           tensor<1x8x512x128xbf16, #kv_full>)
        -> tensor<1x8x512x128xbf16, #q_full>
    return %0 : tensor<1x8x512x128xbf16, #q_full>
  }
}

// CHECK:      "flops": {
// CHECK:        "flops_by_category": {
// CHECK:          "sdpa": 1073741824
// CHECK:        }
// CHECK:        "per_op": [
// CHECK:          "category": "sdpa"
// CHECK:          "flops": 1073741824
// CHECK:          "operation": "ttnn.scaled_dot_product_attention"
// CHECK:        "total_flops": 1073741824
