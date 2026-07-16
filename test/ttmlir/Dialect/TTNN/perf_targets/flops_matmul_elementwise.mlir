// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: ttmlir-opt --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/out.json ttnn-perf-metrics-verbose-output-enabled=true" %s -o /dev/null
// RUN: cat %t.dir/out.json | FileCheck %s

// Per-op FLOP accounting for a matmul followed by two elementwise ops.
//   matmul M=2048, K=128, N=128 -> 2*M*K*N        = 67,108,864 flops
//   relu   2048x128             -> num_elements    =    262,144 flops
//   add    2048x128             -> num_elements    =    262,144 flops
//   total                                          = 67,633,152 flops
// Peak (WH B0, 64 cores, 1.0 GHz): HiFi2 = 64 * 1e9 * 2*32^3 / 32 = 1.31072e14.

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

#act    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#weight = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {ttcore.system_desc = #system_desc} {
  func.func @forward(
      %a: tensor<2048x128xbf16, #act>    {ttcore.argument_type = #ttcore.argument_type<input>},
      %b: tensor<128x128xbf16,  #weight> {ttcore.argument_type = #ttcore.argument_type<parameter>}
  ) -> tensor<2048x128xbf16, #act>
      attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.matmul"(%a, %b) <{transpose_a = false, transpose_b = false}>
        : (tensor<2048x128xbf16, #act>, tensor<128x128xbf16, #weight>)
        -> tensor<2048x128xbf16, #act>
    %1 = "ttnn.relu"(%0) : (tensor<2048x128xbf16, #act>) -> tensor<2048x128xbf16, #act>
    %2 = "ttnn.add"(%1, %0) : (tensor<2048x128xbf16, #act>, tensor<2048x128xbf16, #act>) -> tensor<2048x128xbf16, #act>
    return %2 : tensor<2048x128xbf16, #act>
  }
}

// CHECK:      "flops": {
// CHECK:        "flops_by_category": {
// CHECK-DAG:      "elementwise": 524288
// CHECK-DAG:      "matmul": 67108864
// CHECK:        }
// CHECK:        "peak_flops_per_sec": {
// CHECK:          "hifi2": 131072000000000
// CHECK:        }
// CHECK:        "per_op": [
// CHECK-DAG:      "operation": "ttnn.matmul"
// CHECK-DAG:      "operation": "ttnn.relu"
// CHECK-DAG:      "operation": "ttnn.add"
// CHECK:        "total_flops": 67633152
