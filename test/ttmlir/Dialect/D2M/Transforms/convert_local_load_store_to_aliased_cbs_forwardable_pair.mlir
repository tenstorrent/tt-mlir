// Ensures that a forwardable remote_load/remote_store pair whose store target
// is a local (non-streamed) operand is skipped by the aliased-CB conversion
// pass, so no CB reserve/push/wait/pop ops are emitted.
// RUN: ttmlir-opt --ttcore-register-device --d2m-convert-local-load-store-ops-to-aliased-cbs --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @test_skip_forwardable_local_remote_store_stream_input
// CHECK: d2m.remote_load %{{.*}} %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] %{{.*}}
// CHECK-NOT: d2m.reserve
// CHECK-NOT: d2m.push
// CHECK-NOT: d2m.wait
// CHECK-NOT: d2m.pop

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @test_skip_forwardable_local_remote_store_stream_input(%arg0: memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %stream_in = "d2m.stream_layout"(%arg0, %alloc) <{remapping = #map4}> : (memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) -> memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x3>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream_in : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc_0 : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)  {
    ^unified0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %buffer = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
      %0 = d2m.remote_load %buffer %stream_in[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.view<4>, #dram> -> memref<1x1x!ttcore.tile<32x32, f32>>
      %1 = d2m.remote_store %alloc_0[%c0, %c0] %buffer : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>> -> memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    }
    return
  }
}
