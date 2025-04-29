// RUN: ttmlir-opt --ttir-generic-linearize-memref %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

func.func @add(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>, %arg1: memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: [[cshape0:%[a-z0-9_]+]] = memref.collapse_shape %cb0
    // CHECK-NEXT: [[cshape1:%[a-z0-9_]+]] = memref.collapse_shape %cb1
    // CHECK-NEXT: [[cshape2:%[a-z0-9_]+]] = memref.collapse_shape %cb2
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        // CHECK: = affine.load [[cshape0]][%arg2 * 4 + %arg3]
        // CHECK: = affine.load [[cshape1]][%arg2 * 4 + %arg3]
        %0 = affine.load %cb0[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %1 = affine.load %cb1[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
        // CHECK: affine.store %2, [[cshape2]][%{{.*}} * 4 + %{{.*}}]
        affine.store %2, %cb2[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
      }
    }
    ttir.yield %cb2 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>
}

func.func @addT(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>, %arg1T: memref<1x1x4x2x!tt.tile<32x32, f32>, #tt.shard<8192x4096>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>
  "ttir.generic"(%arg0, %arg1T, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: [[cshape0:%[a-z0-9_]+]] = memref.collapse_shape %cb0
    // CHECK-NEXT: [[cshape1:%[a-z0-9_]+]] = memref.collapse_shape %cb1
    // CHECK-NEXT: [[cshape2:%[a-z0-9_]+]] = memref.collapse_shape %cb2
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        // CHECK: = affine.load [[cshape0]][%arg2 * 4 + %arg3]
        // CHECK: = affine.load [[cshape1]][%arg3 * 2 + %arg2]
        %0 = affine.load %cb0[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %1 = affine.load %cb1[%arg3, %arg2] : memref<4x2x!tt.tile<32x32, f32>, #l1_>
        %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
        // CHECK: affine.store %2, [[cshape2]][%{{.*}} * 4 + %{{.*}}]
        affine.store %2, %cb2[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
      }
    }
    ttir.yield %cb2 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>, memref<1x1x4x2x!tt.tile<32x32, f32>, #tt.shard<8192x4096>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>
}
