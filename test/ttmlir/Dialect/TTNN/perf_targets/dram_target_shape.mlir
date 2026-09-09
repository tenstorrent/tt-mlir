// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: ttmlir-opt --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/out.json" %s -o /dev/null
// RUN: cat %t.dir/out.json | FileCheck %s

// Wormhole B0 chip desc; the forward function has no matmul ops, so the
// matmul-driven roofline / param accumulators all stay at zero. This test
// pins the JSON shape (perf_targets block keys, no kv_cache key).

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

module attributes {ttcore.system_desc = #system_desc} {
  func.func @forward(
      %arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %arg1: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.kv_cache}
  ) -> tensor<32x32xbf16>
      attributes {tt.function_type = "forward_device"} {
    return %arg0 : tensor<32x32xbf16>
  }
}

// CHECK: "perf_targets":
// CHECK: "aiclk_hz":
// CHECK: "arch": "wormhole_b0"
// CHECK: "compute_bound_ops": 0
// CHECK: "dram_bandwidth_bytes_per_sec": 288000000000
// CHECK: "dram_bound_ops": 0
// CHECK: "num_chips": 1
// CHECK: "num_tensix_cores":
// CHECK: "params_count": 0
// CHECK: "params_memory_bytes": 0
// CHECK: "roofline_ms": 0
// CHECK: "skipped_ops": 0
// CHECK: "top_perf_estimate_ms": 0
// CHECK-NOT: "kv_cache":
// CHECK-NOT: "matmul":
// CHECK-NOT: "sdpa_ops":
// CHECK-NOT: "roofline":
