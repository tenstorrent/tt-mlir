// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  memref.global "private" constant @__constant_32x32xf32 : memref<32x32xf32> = dense<1.000000e+00>

  // verify that:
  // - memref addresses get assigned
  // - stream_layouts are emitted and used as generic operands

  func.func @reduce_C(%arg0: memref<1x1x64x96xf32, #ttcore.shard<384x4>, #l1_>) ->memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_> {
    %0 = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
    // CHECK: memref.alloc() {address = {{[^}]+}}} : memref<1x1x1x1x!ttcore.tile
    %alloc = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
    "d2m.to_layout"(%0, %alloc) : (memref<32x32xf32>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) -> ()

    // CHECK: memref.alloc() {address = {{[^}]+}}} : memref<2x3x1x1x!ttcore.tile
    %alloc_0 = memref.alloc() : memref<2x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
    "d2m.to_layout"(%arg0, %alloc_0) : (memref<1x1x64x96xf32, #ttcore.shard<384x4>, #l1_>, memref<2x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) -> ()

    // CHECK: memref.alloc() {address = {{[^}]+}}} : memref<2x1x1x1x!ttcore.tile
    %alloc_1 = memref.alloc() : memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>

    // CHECK: %[[STREAM_A:.+]] = "d2m.stream_layout"(%{{[^}]+}}, %{{[^}]+}}) : (memref<2x3x1x1x!ttcore.tile
    // CHECK: %[[STREAM_B:.+]] = "d2m.stream_layout"(%{{[^}]+}}, %{{[^}]+}}) : (memref<1x1x1x1x!ttcore.tile
    // at the moment, outputs should not be streamed:
    // CHECK-NOT: %[[STREAM_OUT:.+]] = "d2m.stream_layout"(%{{[^}]+}}, %{{[^}]+}}) : (memref<2x1x1x1x!ttcore.tile
    // CHECK: ins(%[[STREAM_A]], %[[STREAM_B]] : memref

    d2m.generic {block_factors = [1, 3], grid = #ttcore.grid<2x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc_0, %alloc : memref<2x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
        outs(%alloc_1 : memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %arg0_unwrap = d2m.pop %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %arg1_unwrap = d2m.pop %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %arg2_unwrap = d2m.reserve %cb2 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0_unwrap, %arg1_unwrap : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) outs(%arg2_unwrap : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_3: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %1 = "d2m.tile_reduce_sum"(%in, %in_3, %out) <{reduce_dim = #d2m<reduce_dim C>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %1 : !ttcore.tile<32x32, f32>
      }
    }

    return %alloc_1 : memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
  }
}
