// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: ttmlir-opt --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/out.json ttnn-perf-metrics-verbose-output-enabled=true" %s -o /dev/null
// RUN: cat %t.dir/out.json | FileCheck %s

// FLOP accounting for the prefill-attention and sparse-matmul formulas. None of
// these ops accepts a compute_config, so all of them are accounted at HiFi4 --
// which is what tt-metal's runtime actually runs a config-less matmul at -- and
// peak_flops_per_sec must come out equal to the hifi4 hardware peak.
//
// Base attention shapes: B=1, Hq=8, Sq=Sk=512, Dk=Dv=128, so the unmasked
// 2*B*Hq*Sq*Sk*(Dk+Dv) = 1,073,741,824.
//   sdpa                      no mask                        = 1,073,741,824
//   sdpa      is_causal       halved (rough)                 =   536,870,912
//   sdpa      sliding_window=128, is_causal: the window caps the effective key
//             length at min(128, 512) and replaces the causal halving, so
//             2*1*8*512*128*256                              =   268,435,456
//   flash_mla_prefill  Q=[1,16,32,128], Sk=32, head_dim_v=64, causal:
//             Dv comes from the attribute, not a V tensor.
//             2*1*16*32*32*(128+64) / 2                      =     3,145,728
//   sparse_matmul  a=[1,1,32,64], b=[1,4,64,32], out E=4 blocks, nnz=2:
//             dense 2*4*32*64*32 = 524,288, scaled by nnz/E   =       262,144
//   total                                                    = 1,882,456,064
//
// Peak (WH B0, 64 cores, 1.0 GHz): hifi4 = 64 * 1e9 * 2*32^3 / 64 = 6.5536e13.

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

// sdpa: [1, 8, 512, 128] -> 4096 rows x 128 cols = 128x4 tiles
#qkv = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 512 + d2, d3), <1x1>, memref<128x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// flash_mla_prefill
#mla_q = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#mla_k = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#mla_o = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// sparse_matmul
#sp_a = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#sp_b = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 64 + d2, d3), <1x1>, memref<8x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#sp_s = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x4xbf16, #dram>, <interleaved>>
#sp_o = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d3 * 32 + d4, d5), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {ttcore.system_desc = #system_desc} {
  func.func @forward(
      %q: tensor<1x8x512x128xbf16, #qkv>     {ttcore.argument_type = #ttcore.argument_type<input>},
      %k: tensor<1x8x512x128xbf16, #qkv>     {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.kv_cache},
      %v: tensor<1x8x512x128xbf16, #qkv>     {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.kv_cache},
      %mq: tensor<1x16x32x128xbf16, #mla_q>  {ttcore.argument_type = #ttcore.argument_type<input>},
      %mk: tensor<1x1x32x128xbf16, #mla_k>   {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.kv_cache},
      %sa: tensor<1x1x32x64xbf16, #sp_a>     {ttcore.argument_type = #ttcore.argument_type<input>},
      %sb: tensor<1x4x64x32xbf16, #sp_b>     {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %ss: tensor<1x1x1x4xbf16, #sp_s>       {ttcore.argument_type = #ttcore.argument_type<input>}
  ) -> tensor<1x1x1x4x32x32xbf16, #sp_o>
      attributes {tt.function_type = "forward_device"} {
    // Unmasked: 2*B*Hq*Sq*Sk*(Dk+Dv) = 1,073,741,824
    %0 = "ttnn.scaled_dot_product_attention"(%q, %k, %v)
        <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x8x512x128xbf16, #qkv>, tensor<1x8x512x128xbf16, #qkv>,
           tensor<1x8x512x128xbf16, #qkv>) -> tensor<1x8x512x128xbf16, #qkv>

    // Causal: halved = 536,870,912
    %1 = "ttnn.scaled_dot_product_attention"(%q, %k, %v)
        <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x8x512x128xbf16, #qkv>, tensor<1x8x512x128xbf16, #qkv>,
           tensor<1x8x512x128xbf16, #qkv>) -> tensor<1x8x512x128xbf16, #qkv>

    // Sliding window caps the effective key length instead of halving = 268,435,456
    %2 = "ttnn.scaled_dot_product_attention"(%q, %k, %v)
        <{is_causal = true, sliding_window_size = 128 : ui32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x8x512x128xbf16, #qkv>, tensor<1x8x512x128xbf16, #qkv>,
           tensor<1x8x512x128xbf16, #qkv>) -> tensor<1x8x512x128xbf16, #qkv>

    // Dv from head_dim_v, causal = 3,145,728
    %3 = "ttnn.flash_mla_prefill"(%mq, %mk)
        <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}>
        : (tensor<1x16x32x128xbf16, #mla_q>, tensor<1x1x32x128xbf16, #mla_k>)
        -> tensor<1x16x32x64xbf16, #mla_o>

    // Dense-equivalent scaled by nnz/E = 262,144
    %4 = "ttnn.sparse_matmul"(%sa, %sb, %ss)
        <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 2 : i64}>
        : (tensor<1x1x32x64xbf16, #sp_a>, tensor<1x4x64x32xbf16, #sp_b>,
           tensor<1x1x1x4xbf16, #sp_s>) -> tensor<1x1x1x4x32x32xbf16, #sp_o>

    return %4 : tensor<1x1x1x4x32x32xbf16, #sp_o>
  }
}

// JSON keys are emitted in alphabetical order.
// CHECK:      "flops": {
// CHECK:        "peak_flops_per_sec": 65536000000000
// CHECK:        "peak_flops_per_sec_by_fidelity": {
// CHECK:        "per_op": [
// CHECK:          "flops": 1073741824
// CHECK:          "math_fidelity": "hifi4"
// CHECK:          "operation": "ttnn.scaled_dot_product_attention"
// CHECK:          "flops": 536870912
// CHECK:          "operation": "ttnn.scaled_dot_product_attention"
// CHECK:          "flops": 268435456
// CHECK:          "operation": "ttnn.scaled_dot_product_attention"
// CHECK:          "flops": 3145728
// CHECK:          "operation": "ttnn.flash_mla_prefill"
// CHECK:          "flops": 262144
// CHECK:          "operation": "ttnn.sparse_matmul"
// CHECK:        ]
// CHECK:        "total_flops": 1882456064
