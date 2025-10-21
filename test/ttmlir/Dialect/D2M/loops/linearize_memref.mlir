// RUN: ttmlir-opt --lower-affine --d2m-generic-linearize-memref -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: #map1 = affine_map<(d0, d1) -> (d0 * 4 + d1)>

func.func @add(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, %arg1: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  "d2m.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
    // CHECK: [[cshape0:%[a-z0-9_]+]] = memref.collapse_shape %cb0
    // CHECK-NEXT: [[cshape1:%[a-z0-9_]+]] = memref.collapse_shape %cb1
    // CHECK-NEXT: [[cshape2:%[a-z0-9_]+]] = memref.collapse_shape %cb2
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        // CHECK: [[APPLY_VAL1:%[a-z0-9_]+]] = affine.apply #map1(%arg2, %arg3)
        // CHECK: %{{.*}} = memref.load [[cshape0]][[[APPLY_VAL1]]]
        // CHECK: [[APPLY_VAL2:%[a-z0-9_]+]] = affine.apply #map1(%arg2, %arg3)
        // CHECK: %{{.*}} = memref.load [[cshape1]][[[APPLY_VAL2]]]
        %0 = affine.load %cb0[%arg2, %arg3] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %1 = affine.load %cb1[%arg2, %arg3] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        // CHECK: [[TILE_ADD:%[a-z0-9_]+]] = "d2m.tile_add"
        %2 = "d2m.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        // CHECK: [[APPLY_VAL3:%[a-z0-9_]+]] = affine.apply #map1(%arg2, %arg3)
        // CHECK: memref.store [[TILE_ADD]], [[cshape2]][[[APPLY_VAL3]]]
        affine.store %2, %cb2[%arg2, %arg3] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      }
    }
    d2m.yield %cb2 : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
}
