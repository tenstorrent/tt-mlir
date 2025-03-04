// RUN: ttmlir-opt --ttir-generic-linearize-memref %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

func.func @add(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), alias>, #l1_>, %arg1: memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), alias>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 2, 0, 1>, operand_cb_mapping = array<i64>}> ({
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: = memref.collapse_shape %arg3
    // CHECK: = memref.collapse_shape %arg4
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        // CHECK: = affine.load %collapse_shape[%arg5 * 4 + %arg6]
        // CHECK: = affine.load %collapse_shape_0[%arg5 * 4 + %arg6]
        %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
        // CHECK: affine.store %2, %collapse_shape_1[%{{.*}} * 4 + %{{.*}}]
        affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
      }
    }
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), alias>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), alias>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
}

func.func @addT(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), alias>, #l1_>, %arg1T: memref<1x1x4x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), alias>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %arg1T, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 2, 0, 1>, operand_cb_mapping = array<i64>}> ({
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: = memref.collapse_shape %arg3
    // CHECK: = memref.collapse_shape %arg4
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        // CHECK: = affine.load %collapse_shape[%arg5 * 4 + %arg6]
        // CHECK: = affine.load %collapse_shape_0[%arg6 * 2 + %arg5]
        %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %1 = affine.load %arg3[%arg6, %arg5] : memref<4x2x!tt.tile<32x32, f32>, #l1_>
        %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
        // CHECK: affine.store %2, %collapse_shape_1[%{{.*}} * 4 + %{{.*}}]
        affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
      }
    }
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), alias>, #l1_>, memref<1x1x4x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), alias>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
}
