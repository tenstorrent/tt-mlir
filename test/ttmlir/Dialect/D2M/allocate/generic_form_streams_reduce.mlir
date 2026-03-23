// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#remap4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  memref.global "private" constant @__constant_32x32xf32 : memref<32x32xf32> = dense<1.000000e+00>

  // verify that:
  // - memref addresses get assigned
  // - CB allocs with CBLayoutAttr are created inside the generic

  func.func @reduce_C(%arg0: memref<1x1x64x96xf32, #ttcore.shard<384x4, 1>, #l1_>) ->memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_> {
    %0 = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
    // CHECK: memref.alloc() {address = {{[^}]+}}} : memref<1x1x1x1x!ttcore.tile
    %alloc = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
    "d2m.to_layout"(%0, %alloc) : (memref<32x32xf32>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) -> ()

    // CHECK: memref.alloc() {address = {{[^}]+}}} : memref<2x3x1x1x!ttcore.tile
    %alloc_0 = memref.alloc() : memref<2x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
    "d2m.to_layout"(%arg0, %alloc_0) : (memref<1x1x64x96xf32, #ttcore.shard<384x4, 1>, #l1_>, memref<2x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) -> ()

    // CHECK: memref.alloc() {address = {{[^}]+}}} : memref<2x1x1x1x!ttcore.tile
    %alloc_1 = memref.alloc() : memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>

    // at the moment, outputs should not be streamed:

    d2m.generic {block_factors = [1, 3], grid = #ttcore.grid<2x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%alloc_0, %alloc : memref<2x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%alloc_1 : memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0():
      %arg0_unwrap = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %arg1_unwrap = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %arg2_unwrap = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      "d2m.tile_matmul_block"(%arg0_unwrap, %arg1_unwrap, %arg2_unwrap) : (memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    }

    return %alloc_1 : memref<2x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
  }
}
