// RUN: ttmlir-opt --tt-register-device --ttir-allocate %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#parallel = #tt.iterator_type<parallel>
#reduction = #tt.iterator_type<reduction>

module {
  memref.global "private" constant @__constant_32x32xf32 : memref<32x32xf32> = dense<1.000000e+00>

  // verify that:
  // - memref addresses get assigned
  // - stream_layouts are emitted and used as generic operands

  func.func @reduce_C(%arg0: memref<1x1x64x96xf32, #tt.shard<384x4>, #l1_>) ->memref<2x1x1x1x!tt.tile<32x32, f32>, #tt.shard<4096x4096>, #l1_> {
    %0 = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
    // CHECK: memref.alloc() {address = {{[^}]+}}} : memref<1x1x1x1x!tt.tile
    %alloc = memref.alloc() : memref<1x1x1x1x!tt.tile<32x32, f32>, #tt.shard<4096x4096>, #l1_>
    "ttir.to_layout"(%0, %alloc) : (memref<32x32xf32>, memref<1x1x1x1x!tt.tile<32x32, f32>, #tt.shard<4096x4096>, #l1_>) -> ()

    // CHECK: memref.alloc() {address = {{[^}]+}}} : memref<2x3x1x1x!tt.tile
    %alloc_0 = memref.alloc() : memref<2x3x1x1x!tt.tile<32x32, f32>, #tt.shard<4096x4096>, #l1_>
    "ttir.to_layout"(%arg0, %alloc_0) : (memref<1x1x64x96xf32, #tt.shard<384x4>, #l1_>, memref<2x3x1x1x!tt.tile<32x32, f32>, #tt.shard<4096x4096>, #l1_>) -> ()

    // CHECK: memref.alloc() {address = {{[^}]+}}} : memref<2x1x1x1x!tt.tile
    %alloc_1 = memref.alloc() : memref<2x1x1x1x!tt.tile<32x32, f32>, #tt.shard<4096x4096>, #l1_>

    // CHECK: %[[STREAM_A:.+]] = "ttir.stream_layout"(%{{[^}]+}}, %{{[^}]+}}) : (memref<2x3x1x1x!tt.tile
    // CHECK: %[[STREAM_B:.+]] = "ttir.stream_layout"(%{{[^}]+}}, %{{[^}]+}}) : (memref<1x1x1x1x!tt.tile
    // CHECK: ins(%[[STREAM_A]], %[[STREAM_B]] : memref

    ttir.generic {block_factors = [1, 1], grid = #tt.grid<2x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #reduction], threads = [#ttir.thread<compute>]}
        ins(%alloc_0, %alloc : memref<2x3x1x1x!tt.tile<32x32, f32>, #tt.shard<4096x4096>, #l1_>, memref<1x1x1x1x!tt.tile<32x32, f32>, #tt.shard<4096x4096>, #l1_>)
        outs(%alloc_1 : memref<2x1x1x1x!tt.tile<32x32, f32>, #tt.shard<4096x4096>, #l1_>)  {
    ^compute0(%cb0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<1x1x!tt.tile<32x32, f32>, #l1_>):
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%cb0, %cb1 : memref<1x1x!tt.tile<32x32, f32>, #l1_>, memref<1x1x!tt.tile<32x32, f32>, #l1_>) outs(%cb2 : memref<1x1x!tt.tile<32x32, f32>, #l1_>) {
      ^bb0(%in: !tt.tile<32x32, f32>, %in_3: !tt.tile<32x32, f32>, %out: !tt.tile<32x32, f32>):
        %1 = "ttir.tile_reduce_sum"(%in, %in_3) <{reduce_dim = #ttir<reduce_dim C>}> : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
        linalg.yield %1 : !tt.tile<32x32, f32>
      }
    }

    return %alloc_1 : memref<2x1x1x1x!tt.tile<32x32, f32>, #tt.shard<4096x4096>, #l1_>
  }
}
