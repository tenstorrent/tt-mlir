// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: ttmlir-opt --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/out.json" %s -o /dev/null
// RUN: cat %t.dir/out.json | FileCheck %s

// Exercises the TileType branch of getBytesPerScalarElement. Param + KV
// args carry !ttcore.tile<32x32, bfp_bf8> as their element type, so the
// per-scalar byte size comes out of tile.getSizeBytes() / tileElements
// instead of getIntOrFloatBitWidth() / 8. Expected values:
//   - bfp_bf8 tile is 1088 B per 32x32 tile (1024 mantissa + 64 exponent),
//     so memory_bytes uses 1088/1024 = 1.0625 B/scalar.
//   - memory_bytes_bfp8 uses our canonical 1056/1024 = 1.03125 B/scalar
//     constant from the pass (slightly lower; matches the parent-branch
//     calibration of the DRAM roofline).
//   - params: 64x64 = 4096 scalars -> memory_bytes 4352, _bfp8 4224.
//   - kv:     32x32 = 1024 scalars -> memory_bytes 1088, _bfp8 1056.

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
      %arg1: tensor<64x64x!ttcore.tile<32x32, bfp_bf8>> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %arg2: tensor<32x32x!ttcore.tile<32x32, bfp_bf8>> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.kv_cache}
  ) -> tensor<32x32xbf16>
      attributes {tt.function_type = "forward_device"} {
    return %arg0 : tensor<32x32xbf16>
  }
}

// CHECK: "perf_targets":
// CHECK: "arch": "wormhole_b0"
// CHECK: "dram_bandwidth_bytes_per_sec": 288000000000
// CHECK: "kv_cache":
// CHECK: "count": 1024
// CHECK: "memory_bytes": 1088
// CHECK: "memory_bytes_bfp8": 1056
// CHECK: "num_chips": 1
// CHECK: "params":
// CHECK: "count": 4096
// CHECK: "memory_bytes": 4352
// CHECK: "memory_bytes_bfp8": 4224
// CHECK: "roofline":
// CHECK: "dram_time_ms":
// CHECK: "top_perf_samples_per_sec":
// CHECK: "top_perf_time_ms":
