// UNSUPPORTED: true
// this test requires proper CB buffer sizing logic
//
// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate=allow-output-spilling=1 -o %t %s
// RUN: FileCheck %s --input-file=%t

// This will succeed after spilling some (but not all) allocs to DRAM.
// Note: `allocate.spill_negative.mlir` is a negative version of this test.

// CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #l1>
// CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #dram>

#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 3072x3072, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
module {
  func.func @main(%arg0: memref<3072x3072xf32>) -> memref<3072x3072xf32> {
    %alloc = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    %alloc_0 = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_0 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<12x12x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_1 = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_0 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_1 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<12x12x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_2 = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_1 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_2 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<12x12x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_3 = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_2 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_3 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<12x12x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_4 = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_3 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_4 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<12x12x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_5 = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_4 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_5 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<12x12x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) outs(%cb1 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_cos"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_6 = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc_3, %alloc_5 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>, memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_6 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb2: memref<12x12x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>, memref<12x12x!ttcore.tile<32x32, f32>, #l1>) outs(%cb2 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_12: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%in, %in_12, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_7 = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc_2, %alloc_6 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>, memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_7 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb2: memref<12x12x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>, memref<12x12x!ttcore.tile<32x32, f32>, #l1>) outs(%cb2 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_12: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%in, %in_12, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_8 = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc_1, %alloc_7 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>, memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_8 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb2: memref<12x12x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>, memref<12x12x!ttcore.tile<32x32, f32>, #l1>) outs(%cb2 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_12: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%in, %in_12, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_9 = memref.alloc() : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc_0, %alloc_8 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>, memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_9 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb2: memref<12x12x!ttcore.tile<32x32, f32>, #l1>):
      linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>, memref<12x12x!ttcore.tile<32x32, f32>, #l1>) outs(%cb2 : memref<12x12x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_12: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%in, %in_12, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    %alloc_10 = memref.alloc() : memref<3072x3072xf32>
    %alloc_11 = memref.alloc() : memref<8x8x384x384xf32, #ttcore.shard<1536x4>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc_9 : memref<8x8x12x12x!ttcore.tile<32x32, f32>, #ttcore.shard<49152x4096>, #l1>)
        outs(%alloc_11 : memref<8x8x384x384xf32, #ttcore.shard<1536x4>, #l1>)  {
    ^compute0(%cb0: memref<12x12x!ttcore.tile<32x32, f32>, #l1>, %cb1: memref<384x384xf32, #l1>):
      "d2m.tile_untilize_block"(%cb0, %cb1) : (memref<12x12x!ttcore.tile<32x32, f32>, #l1>, memref<384x384xf32, #l1>) -> ()
    }
    d2m.to_layout %alloc_11, %alloc_10 : memref<8x8x384x384xf32, #ttcore.shard<1536x4>, #l1> into memref<3072x3072xf32> hostInfo = #layout
    return %alloc_10 : memref<3072x3072xf32>
  }
}
