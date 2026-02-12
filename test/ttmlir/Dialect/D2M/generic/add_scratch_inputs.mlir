// RUN: ttmlir-opt --ttcore-register-device --d2m-add-scratch-inputs %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#parallel = #ttcore.iterator_type<parallel>

!tile_f32 = !ttcore.tile<32x32, f32>
!memref_tiled = memref<1x1x4x4x!tile_f32, #ttcore.shard<16384x4096, 1>, #l1>
!cb = !d2m.cb<memref<4x4x!tile_f32, #l1>>

#map = affine_map<(d0, d1) -> (d0, d1)>

// Two tile_add ops in a fused generic → scratch should be added at input index 2.
// Verify: scratch memref.alloc (1x1x1x32 for f32 @ 128KB) and scratch_inputs attr.

// CHECK-LABEL: func.func @two_adds_gets_scratch
// CHECK: memref.alloc() : memref<1x1x1x32x!ttcore.tile<32x32, f32>
// CHECK: d2m.generic
// CHECK-SAME: scratch_inputs = array<i64: 2>
// CHECK: ins(%{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-SAME: memref<1x1x1x32x!ttcore.tile<32x32, f32>
func.func @two_adds_gets_scratch(%arg0: !memref_tiled, %arg1: !memref_tiled) {
  %out = memref.alloc() : !memref_tiled
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [#map, #map, #map],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%arg0, %arg1 : !memref_tiled, !memref_tiled)
  outs(%out : !memref_tiled) {
  ^unified0(%cb0: !cb, %cb1: !cb, %cb2: !cb):
    %block0 = d2m.block_index(0) : index
    %block1 = d2m.block_index(1) : index
    %e0 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    %a = d2m.remote_load %e0 %arg0[%block0, %block1] : memref<4x4x!tile_f32>, !memref_tiled -> memref<4x4x!tile_f32, #l1>
    %e1 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    %b = d2m.remote_load %e1 %arg1[%block0, %block1] : memref<4x4x!tile_f32>, !memref_tiled -> memref<4x4x!tile_f32, #l1>
    // First add: tmp = a + b.
    %e2 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%e0, %e1 : memref<4x4x!tile_f32>, memref<4x4x!tile_f32>) outs(%e2 : memref<4x4x!tile_f32>) {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_add"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    }
    // Second add: result = tmp + a.
    %e3 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%e2, %e0 : memref<4x4x!tile_f32>, memref<4x4x!tile_f32>) outs(%e3 : memref<4x4x!tile_f32>) {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_add"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    }
    %stored = d2m.remote_store %out[%block0, %block1] %e3 : !memref_tiled, memref<4x4x!tile_f32> -> !memref_tiled
  }
  return
}

// Mixed binary FPU ops (tile_add + tile_mul) → scratch should be added.

// CHECK-LABEL: func.func @add_and_mul_gets_scratch
// CHECK: memref.alloc() : memref<1x1x1x32x!ttcore.tile<32x32, f32>
// CHECK: d2m.generic
// CHECK-SAME: scratch_inputs = array<i64: 2>
// CHECK: ins(%{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-SAME: memref<1x1x1x32x!ttcore.tile<32x32, f32>
func.func @add_and_mul_gets_scratch(%arg0: !memref_tiled, %arg1: !memref_tiled) {
  %out = memref.alloc() : !memref_tiled
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [#map, #map, #map],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%arg0, %arg1 : !memref_tiled, !memref_tiled)
  outs(%out : !memref_tiled) {
  ^unified0(%cb0: !cb, %cb1: !cb, %cb2: !cb):
    %block0 = d2m.block_index(0) : index
    %block1 = d2m.block_index(1) : index
    %e0 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    %a = d2m.remote_load %e0 %arg0[%block0, %block1] : memref<4x4x!tile_f32>, !memref_tiled -> memref<4x4x!tile_f32, #l1>
    %e1 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    %b = d2m.remote_load %e1 %arg1[%block0, %block1] : memref<4x4x!tile_f32>, !memref_tiled -> memref<4x4x!tile_f32, #l1>
    // Add: tmp = a + b.
    %e2 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%e0, %e1 : memref<4x4x!tile_f32>, memref<4x4x!tile_f32>) outs(%e2 : memref<4x4x!tile_f32>) {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_add"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    }
    // Mul: result = tmp * a.
    %e3 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%e2, %e0 : memref<4x4x!tile_f32>, memref<4x4x!tile_f32>) outs(%e3 : memref<4x4x!tile_f32>) {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_mul"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    }
    %stored = d2m.remote_store %out[%block0, %block1] %e3 : !memref_tiled, memref<4x4x!tile_f32> -> !memref_tiled
  }
  return
}

// Single tile_add → no scratch (needsScratch requires > 1 binary FPU op).

// CHECK-LABEL: func.func @single_add_no_scratch
// CHECK-NOT: scratch_inputs
// CHECK: return
func.func @single_add_no_scratch(%arg0: !memref_tiled, %arg1: !memref_tiled) {
  %out = memref.alloc() : !memref_tiled
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [#map, #map, #map],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%arg0, %arg1 : !memref_tiled, !memref_tiled)
  outs(%out : !memref_tiled) {
  ^unified0(%cb0: !cb, %cb1: !cb, %cb2: !cb):
    %block0 = d2m.block_index(0) : index
    %block1 = d2m.block_index(1) : index
    %e0 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    %a = d2m.remote_load %e0 %arg0[%block0, %block1] : memref<4x4x!tile_f32>, !memref_tiled -> memref<4x4x!tile_f32, #l1>
    %e1 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    %b = d2m.remote_load %e1 %arg1[%block0, %block1] : memref<4x4x!tile_f32>, !memref_tiled -> memref<4x4x!tile_f32, #l1>
    %e2 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%e0, %e1 : memref<4x4x!tile_f32>, memref<4x4x!tile_f32>) outs(%e2 : memref<4x4x!tile_f32>) {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_add"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    }
    %stored = d2m.remote_store %out[%block0, %block1] %e2 : !memref_tiled, memref<4x4x!tile_f32> -> !memref_tiled
  }
  return
}
