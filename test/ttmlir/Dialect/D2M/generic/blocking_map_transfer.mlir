// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-scratch-buffers --d2m-generic-apply-interchange --d2m-generate-outer-loops %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#parallel = #ttcore.iterator_type<parallel>

!tile_f32 = !ttcore.tile<32x32, f32>
!memref_tiled = memref<1x1x4x4x!tile_f32, #ttcore.shard<16384x4096, 1>, #l1>

#map = affine_map<(d0, d1) -> (d0, d1)>

// d2m.blocking_map on inner linalg ops (set by elementwise fusion) should
// transfer to the output memref.alloc in InsertScratchBuffers and survive
// through ApplyInterchange + GenerateOuterLoops down to the allocator.

// CHECK-LABEL: func.func @blocking_map_transfers_to_alloc
// CHECK: d2m.generic
// CHECK: memref.alloc() {alignment = 64 : i64, d2m.blocking_map = #map} : memref<4x4x!ttcore.tile<32x32, f32>>
func.func @blocking_map_transfers_to_alloc(%arg0: !memref_tiled, %arg1: !memref_tiled) {
  %out = memref.alloc() : !memref_tiled
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [#map, #map, #map],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%arg0, %arg1 : !memref_tiled, !memref_tiled)
  outs(%out : !memref_tiled) {
  ^unified0:
    %block0 = d2m.block_index(0) : index
    %block1 = d2m.block_index(1) : index
    %e0 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    %a = d2m.remote_load %e0 %arg0[%block0, %block1] : memref<4x4x!tile_f32>, !memref_tiled -> memref<4x4x!tile_f32, #l1>
    %e1 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    %b = d2m.remote_load %e1 %arg1[%block0, %block1] : memref<4x4x!tile_f32>, !memref_tiled -> memref<4x4x!tile_f32, #l1>
    %intermediate = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%e0, %e1 : memref<4x4x!tile_f32>, memref<4x4x!tile_f32>)
    outs(%intermediate : memref<4x4x!tile_f32>)
    attrs = {d2m.blocking_map = #map} {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_add"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    }
    %e3 = memref.alloc() {alignment = 64 : i64} : memref<4x4x!tile_f32>
    linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%intermediate, %e0 : memref<4x4x!tile_f32>, memref<4x4x!tile_f32>)
    outs(%e3 : memref<4x4x!tile_f32>)
    attrs = {d2m.blocking_map = #map} {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_add"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    }
    %stored = d2m.remote_store %out[%block0, %block1] %e3 : !memref_tiled, memref<4x4x!tile_f32> -> !memref_tiled
  }
  return
}
