// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=test-assume-l1-capacity=393216 allow-l1-output-spilling=1" -o %t %s
// RUN: FileCheck %s --input-file=%t

// This will succeed after spilling some (but not all) allocs to DRAM.
// Note: `allocate.spill_negative.mlir` is a negative version of this test.

// CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #l1>
// CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #dram>

#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @main(%arg0: memref<1024x1024xf32>) -> memref<1024x1024xf32> {
    %alloc = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    %alloc_0 = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_0 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_1 = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_0 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_1 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_2 = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_1 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_2 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_3 = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_2 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_3 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_4 = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_3 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_4 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_5 = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_4 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_5 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_6 = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    d2m.generic {block_factors = [1, 1, 8], grid = #ttcore.grid<8x8>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc_3, %alloc_5 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_6 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb2: memref<4x4x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%cb2 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_12: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%in, %in_12, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_7 = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    d2m.generic {block_factors = [1, 1, 8], grid = #ttcore.grid<8x8>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc_2, %alloc_6 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_7 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb2: memref<4x4x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%cb2 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_12: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%in, %in_12, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_8 = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    d2m.generic {block_factors = [1, 1, 8], grid = #ttcore.grid<8x8>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc_1, %alloc_7 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_8 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb2: memref<4x4x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%cb2 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_12: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%in, %in_12, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_9 = memref.alloc() : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
    d2m.generic {block_factors = [1, 1, 8], grid = #ttcore.grid<8x8>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc_0, %alloc_8 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_9 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb2: memref<4x4x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) outs(%cb2 : memref<4x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_12: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%in, %in_12, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_10 = memref.alloc() : memref<1024x1024xf32>
    %alloc_11 = memref.alloc() : memref<8x8x128x128xf32, #ttcore.shard<512x4>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_9 : memref<8x8x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>)
        outs(%alloc_11 : memref<8x8x128x128xf32, #ttcore.shard<512x4>, #l1>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<128x128xf32, #l1>):
      "d2m.tile_untilize_block"(%cb0, %cb1) : (memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<128x128xf32, #l1>) -> ()
    }
    d2m.to_layout %alloc_11, %alloc_10 : memref<8x8x128x128xf32, #ttcore.shard<512x4>, #l1> into memref<1024x1024xf32> hostInfo = #layout
    return %alloc_10 : memref<1024x1024xf32>
  }
}
