// RUN: ttmlir-opt --ttcore-register-device --d2m-convert-local-load-store-ops-to-aliased-cbs --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @test_convert_local_load_store
// CHECK-DAG: %[[CB0:.*]] = d2m.get_cb(0)
// CHECK-DAG: %[[CB1:.*]] = d2m.get_cb(1)
// CHECK-DAG: %[[CB2:.*]] = d2m.get_cb(2)
// CHECK-DAG: d2m.reserve %[[CB2]]
// CHECK: d2m.reserve %[[CB0]]
// CHECK: d2m.push %[[CB0]]
// CHECK: d2m.wait %[[CB0]]
// CHECK: d2m.reserve %[[CB1]]
// CHECK: d2m.push %[[CB1]]
// CHECK: d2m.wait %[[CB1]]
// CHECK: linalg.generic
// CHECK: d2m.tile_add
// CHECK: d2m.pop %[[CB1]]
// CHECK: d2m.pop %[[CB0]]
// CHECK: d2m.push %[[CB2]]
// CHECK: d2m.wait %[[CB2]]
// CHECK: d2m.pop %[[CB2]]
// CHECK-NOT: d2m.remote_load
// CHECK-NOT: d2m.remote_store

#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 128x96, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @test_convert_local_load_store() {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x3>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc, %alloc_0 : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%alloc_1 : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)  {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          %buffer0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
          %buffer1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
          %buffer = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
          d2m.remote_load %buffer0 %alloc[%0, %1] : memref<1x1x!ttcore.tile<32x32, f32>>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>>
          d2m.remote_load %buffer1 %alloc_0[%0, %1] : memref<1x1x!ttcore.tile<32x32, f32>>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>>

          linalg.generic {
            indexing_maps = [#map, #map, #map],
            iterator_types = ["parallel", "parallel"]
          } ins(%buffer0, %buffer1 : memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x!ttcore.tile<32x32, f32>>)
            outs(%buffer : memref<1x1x!ttcore.tile<32x32, f32>>) {
          ^bb0(%in0: !ttcore.tile<32x32, f32>, %in1: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %add = "d2m.tile_add"(%in0, %in1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %add : !ttcore.tile<32x32, f32>
          }

          %result = d2m.remote_store %alloc_1[%0, %1] %buffer : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>> -> memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // CHECK-LABEL: func.func @test_remote_load_alias_view_last_use
  // CHECK-DAG: %[[CB0:.*]] = d2m.get_cb(0)
  // CHECK: d2m.reserve %[[CB0]]
  // CHECK: d2m.push %[[CB0]]
  // CHECK: %[[WAIT:.*]] = d2m.wait %[[CB0]]
  // CHECK: %[[VIEW:.*]] = memref.collapse_shape %[[WAIT]] {{.*}} : memref<2x2x!ttcore.tile<32x32, f32>, #l1> into memref<4x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: %[[TILE:.*]] = memref.load %[[VIEW]][%{{.*}}] : memref<4x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: d2m.pop %[[CB0]]
  // CHECK: memref.store %[[TILE]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1>
  // CHECK-NOT: d2m.remote_load
  func.func @test_remote_load_alias_view_last_use() {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1>)
        outs(%alloc_0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %buffer = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      d2m.remote_load %buffer %alloc[%c0, %c0] : memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %view = memref.collapse_shape %buffer [[0, 1]] : memref<2x2x!ttcore.tile<32x32, f32>, #l1> into memref<4x!ttcore.tile<32x32, f32>, #l1>
      %tile = memref.load %view[%c0] : memref<4x!ttcore.tile<32x32, f32>, #l1>
      memref.store %tile, %alloc_0[%c0, %c0, %c0, %c0] : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1>
    }
    return
  }

  // CHECK-LABEL: func.func @test_skip_local_remote_load_with_multicast
  // CHECK: d2m.remote_load %{{.*}} %{{.*}}[%{{.*}}, %{{.*}}] mcore[%{{.*}}, %{{.*}}] mshape[%{{.*}}, %{{.*}}]
  // CHECK-NOT: d2m.reserve
  // CHECK-NOT: d2m.push
  // CHECK-NOT: d2m.wait
  // CHECK-NOT: d2m.pop
  func.func @test_skip_local_remote_load_with_multicast() {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x16384, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x16384, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc : memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x16384, 1>, #l1>)
        outs(%alloc_0 : memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x16384, 1>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %buffer = memref.alloc() : memref<1x4x!ttcore.tile<32x32, f32>>
      %sink = memref.alloc() : memref<1x4x!ttcore.tile<32x32, f32>>
      %in = d2m.remote_load %buffer %alloc[%c0, %c0] mcore[%c0, %c0] mshape[%c1, %c4] : memref<1x4x!ttcore.tile<32x32, f32>>, memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x16384, 1>, #l1> -> memref<1x4x!ttcore.tile<32x32, f32>>
      %tile = memref.load %in[%c0, %c0] : memref<1x4x!ttcore.tile<32x32, f32>>
      memref.store %tile, %sink[%c0, %c0] : memref<1x4x!ttcore.tile<32x32, f32>>
    }
    return
  }

}
