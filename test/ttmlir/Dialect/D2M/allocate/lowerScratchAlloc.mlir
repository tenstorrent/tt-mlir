// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-scratch-allocate %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#parallel = #ttcore.iterator_type<parallel>

module {

// --- Test 1: Single scratch_allocate replaced by subview of scratch CB ---

// CHECK-LABEL: func.func @single_scratch_store_load
func.func @single_scratch_store_load() {
  %in = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  %scratch_buf = memref.alloc() : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
  %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    scratch_inputs = array<i64: 1>,
    threads = [#d2m.thread<unified>]
  }
  ins(%in, %scratch_buf : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>)
  outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^bb0():
    %alloc_cb0 = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb1 = memref.alloc() : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb2 = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index
    // CHECK: d2m.get_cb(1)
    // CHECK: %[[SCRATCH:.*]] = d2m.get_scratch_from_cb %{{.*}}
    // CHECK: %[[SV:.*]] = memref.subview %[[SCRATCH]][0, 0] [1, 1] [1, 1]
    // CHECK-SAME: to memref<1x!ttcore.tile<32x32, f32>
    %scratch_slot = d2m.scratch_allocate {slot = 0 : i64} : memref<1x!ttcore.tile<32x32, f32>, #l1>
    // Verify store and load are rewritten to use the subview.
    // CHECK: memref.load %[[SV]][%{{.*}}]
    // CHECK: memref.store %{{.*}}, %[[SV]][%{{.*}}]
    %tile = memref.load %scratch_slot[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
    memref.store %tile, %scratch_slot[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
  }
  // CHECK-NOT: d2m.scratch_allocate
  return
}

// --- Test 2: Two conflicting scratch_allocates ---
// Both slots are alive at the same time, so they cannot be packed together.
// Sorted by size descending: slot 1 (2 tiles) then slot 0 (1 tile).
// slot 1 -> offset 0, slot 0 -> offset 2.

// CHECK-LABEL: func.func @two_scratch_conflicting
func.func @two_scratch_conflicting() {
  %in = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  %scratch_buf = memref.alloc() : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
  %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    scratch_inputs = array<i64: 1>,
    threads = [#d2m.thread<unified>]
  }
  ins(%in, %scratch_buf : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>)
  outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^bb0():
    %alloc_cb3 = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb4 = memref.alloc() : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb5 = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index
    // CHECK: d2m.get_cb(1)
    // CHECK: %[[SCRATCH:.*]] = d2m.get_scratch_from_cb %{{.*}}

    // Slot 0 (1 tile): lives [1, 4]. Conflicts with slot 1 [2, 5].
    // Sorted by size: slot 1 gets offset 0, slot 0 gets offset 2.
    // CHECK: %[[SV0:.*]] = memref.subview %[[SCRATCH]][0, 2] [1, 1] [1, 1]
    // CHECK-SAME: to memref<1x!ttcore.tile<32x32, f32>
    %s0 = d2m.scratch_allocate {slot = 0 : i64} : memref<1x!ttcore.tile<32x32, f32>, #l1>

    // Slot 1 (2 tiles): lives [2, 5]. Conflicts with slot 0 [1, 4].
    // CHECK: %[[SV1:.*]] = memref.subview %[[SCRATCH]][0, 0] [1, 2] [1, 1]
    // CHECK-SAME: to memref<2x!ttcore.tile<32x32, f32>
    %s1 = d2m.scratch_allocate {slot = 1 : i64} : memref<2x!ttcore.tile<32x32, f32>, #l1>

    %tile = memref.load %s0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
    memref.store %tile, %s0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
    memref.store %tile, %s1[%c0] : memref<2x!ttcore.tile<32x32, f32>, #l1>
  }
  // CHECK-NOT: d2m.scratch_allocate
  return
}

// --- Test 3: Packing non-conflicting allocations ---
// slot 2 (5 tiles) is defined and last-used early, then goes dead.
// slots 0 and 1 are alive later. They are packed inside slot 2's footprint,
// reducing peak usage from 8 to 5.
// slot 2 -> offset 0 (outer), slot 1 -> offset 0 (packed), slot 0 -> offset 2 (packed)

// CHECK-LABEL: func.func @packing_non_conflicting
func.func @packing_non_conflicting() {
  %in = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  %scratch_buf = memref.alloc() : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
  %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    scratch_inputs = array<i64: 1>,
    threads = [#d2m.thread<unified>]
  }
  ins(%in, %scratch_buf : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>)
  outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^bb0():
    %alloc_cb11 = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb12 = memref.alloc() : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb13 = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index
    // CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<1x8x!ttcore.tile<32x32, f32>,

    // Slot 2 (5 tiles): defined and last-used here. Dies before slots 0/1.
    // CHECK: %[[SV2:.*]] = memref.subview %[[SCRATCH]][0, 0] [1, 5] [1, 1]
    // CHECK-SAME: to memref<5x!ttcore.tile<32x32, f32>
    %s2 = d2m.scratch_allocate {slot = 2 : i64} : memref<5x!ttcore.tile<32x32, f32>, #l1>
    %tile_early = memref.load %s2[%c0] : memref<5x!ttcore.tile<32x32, f32>, #l1>

    // Slot 0 (1 tile): packed inside slot 2 at offset 2.
    // CHECK: %[[SV0:.*]] = memref.subview %[[SCRATCH]][0, 2] [1, 1] [1, 1]
    // CHECK-SAME: to memref<1x!ttcore.tile<32x32, f32>
    %s0 = d2m.scratch_allocate {slot = 0 : i64} : memref<1x!ttcore.tile<32x32, f32>, #l1>

    // Slot 1 (2 tiles): packed inside slot 2 at offset 0.
    // CHECK: %[[SV1:.*]] = memref.subview %[[SCRATCH]][0, 0] [1, 2] [1, 1]
    // CHECK-SAME: to memref<2x!ttcore.tile<32x32, f32>
    %s1 = d2m.scratch_allocate {slot = 1 : i64} : memref<2x!ttcore.tile<32x32, f32>, #l1>

    %tile = memref.load %s0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
    memref.store %tile, %s0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
    memref.store %tile, %s1[%c0] : memref<2x!ttcore.tile<32x32, f32>, #l1>
  }
  // CHECK-NOT: d2m.scratch_allocate
  return
}

// --- Test 4: Generic without scratch_inputs is unchanged ---

// CHECK-LABEL: func.func @no_scratch_noop
func.func @no_scratch_noop() {
  %in = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  // CHECK-NOT: d2m.get_scratch_from_cb
  // CHECK-NOT: memref.subview
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%in : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
  outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^bb0():
    %alloc_cb6 = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb7 = memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
  }
  return
}

// --- Test 5: Scratch with bf16 tiles ---

// CHECK-LABEL: func.func @scratch_bf16
func.func @scratch_bf16() {
  %in = memref.alloc() : memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #l1>
  %scratch_buf = memref.alloc() : memref<1x1x1x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048, 1>, #l1>
  %out = memref.alloc() : memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #l1>
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    scratch_inputs = array<i64: 1>,
    threads = [#d2m.thread<unified>]
  }
  ins(%in, %scratch_buf : memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #l1>, memref<1x1x1x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048, 1>, #l1>)
  outs(%out : memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #l1>) {
  ^bb0():
    %alloc_cb8 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, bf16>, #l1>
    %alloc_cb9 = memref.alloc() : memref<1x4x!ttcore.tile<32x32, bf16>, #l1>
    %alloc_cb10 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, bf16>, #l1>
    %c0 = arith.constant 0 : index
    // CHECK: d2m.get_cb(1)
    // CHECK: %[[SCRATCH:.*]] = d2m.get_scratch_from_cb %{{.*}}
    // CHECK: memref.subview %[[SCRATCH]][0, 0] [1, 1] [1, 1]
    // CHECK-SAME: to memref<1x!ttcore.tile<32x32, bf16>
    %scratch_slot = d2m.scratch_allocate {slot = 0 : i64} : memref<1x!ttcore.tile<32x32, bf16>, #l1>
    %tile = memref.load %scratch_slot[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1>
    memref.store %tile, %scratch_slot[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1>
  }
  // CHECK-NOT: d2m.scratch_allocate
  return
}

}
