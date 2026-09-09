// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-scratch-allocate %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#parallel = #ttcore.iterator_type<parallel>

module {

// --- Test 1: Single scratch_allocate replaced by subview of scratch memref ---

// CHECK-LABEL: func.func @single_scratch_store_load
func.func @single_scratch_store_load() {
  %in = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%in : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
  outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^bb0():
    %alloc_cb0 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb1 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    // CHECK: %[[SCRATCH:.*]] = memref.alloc() {d2m.scratch_buffer} : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NOT: d2m.scratch_init
    %scratch = memref.alloc() {d2m.scratch_buffer} : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    d2m.scratch_init %scratch : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index
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
// Both slots are written then read with interleaved use ranges, so their live
// ranges genuinely overlap and they cannot be packed together.
// Sorted by size descending: slot 1 (2 tiles) then slot 0 (1 tile).
// slot 1 -> offset 0, slot 0 -> offset 2.

// CHECK-LABEL: func.func @two_scratch_conflicting
func.func @two_scratch_conflicting() {
  %in = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%in : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
  outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^bb0():
    %alloc_cb3 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb4 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    // CHECK: %[[SCRATCH:.*]] = memref.alloc() {d2m.scratch_buffer} : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NOT: d2m.scratch_init
    %scratch = memref.alloc() {d2m.scratch_buffer} : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    d2m.scratch_init %scratch : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index
    %seed = memref.load %alloc_cb3[%c0, %c0] : memref<4x4x!ttcore.tile<32x32, f32>, #l1>

    // Sorted by size descending: slot 1 (2 tiles) gets offset 0, slot 0
    // (1 tile) cannot pack inside it because their use spans interleave
    // (store s0, store s1, load s0, load s1) and gets offset 2.
    // CHECK: %[[SV0:.*]] = memref.subview %[[SCRATCH]][0, 2] [1, 1] [1, 1]
    // CHECK-SAME: to memref<1x!ttcore.tile<32x32, f32>
    %s0 = d2m.scratch_allocate {slot = 0 : i64} : memref<1x!ttcore.tile<32x32, f32>, #l1>

    // CHECK: %[[SV1:.*]] = memref.subview %[[SCRATCH]][0, 0] [1, 2] [1, 1]
    // CHECK-SAME: to memref<2x!ttcore.tile<32x32, f32>
    %s1 = d2m.scratch_allocate {slot = 1 : i64} : memref<2x!ttcore.tile<32x32, f32>, #l1>

    memref.store %seed, %s0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
    memref.store %seed, %s1[%c0] : memref<2x!ttcore.tile<32x32, f32>, #l1>
    %t0 = memref.load %s0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
    %t1 = memref.load %s1[%c0] : memref<2x!ttcore.tile<32x32, f32>, #l1>
    memref.store %t1, %alloc_cb4[%c0, %c0] : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    memref.store %t0, %alloc_cb4[%c0, %c0] : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
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
  %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%in : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
  outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^bb0():
    %alloc_cb11 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb12 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    // CHECK: %[[SCRATCH:.*]] = memref.alloc() {d2m.scratch_buffer} : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NOT: d2m.scratch_init
    %scratch = memref.alloc() {d2m.scratch_buffer} : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    d2m.scratch_init %scratch : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index

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

// --- Test 4: Allocation between in-place read and write ---
// Slot 0 is read, then slot 1 is allocated and used before slot 0 is written
// in place. Slot 0's last store happens after slot 1's only use, so under
// use-based liveness the two equal-sized slots overlap and cannot share
// storage. Tie-breaking by slot id makes slot 0 offset 0 and slot 1 offset 2.

// CHECK-LABEL: func.func @allocation_between_in_place_read_write
func.func @allocation_between_in_place_read_write() {
  %in = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%in : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
  outs(%out : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^bb0():
    %alloc_cb13 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    // CHECK: %[[SCRATCH:.*]] = memref.alloc() {d2m.scratch_buffer} : memref<1x4x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NOT: d2m.scratch_init
    %scratch = memref.alloc() {d2m.scratch_buffer} : memref<1x4x!ttcore.tile<32x32, f32>, #l1>
    d2m.scratch_init %scratch : memref<1x4x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index

    // CHECK: %[[SV0:.*]] = memref.subview %[[SCRATCH]][0, 0] [1, 2] [1, 1]
    // CHECK-SAME: to memref<2x!ttcore.tile<32x32, f32>
    %s0 = d2m.scratch_allocate {slot = 0 : i64} : memref<2x!ttcore.tile<32x32, f32>, #l1>

    %producer_value = memref.load %alloc_cb13[%c0, %c0] : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    // CHECK: memref.store %{{.*}}, %[[SV0]][%{{.*}}]
    memref.store %producer_value, %s0[%c0] : memref<2x!ttcore.tile<32x32, f32>, #l1>

    // CHECK: %[[TILE:.*]] = memref.load %[[SV0]][%{{.*}}]
    %tile = memref.load %s0[%c0] : memref<2x!ttcore.tile<32x32, f32>, #l1>

    // CHECK: %[[SV1:.*]] = memref.subview %[[SCRATCH]][0, 2] [1, 2] [1, 1]
    // CHECK-SAME: to memref<2x!ttcore.tile<32x32, f32>
    %s1 = d2m.scratch_allocate {slot = 1 : i64} : memref<2x!ttcore.tile<32x32, f32>, #l1>

    // s1 is stored to first, then s0 receives its in-place write, so
    // s0's live range strictly contains s1's.
    // CHECK: memref.store %[[TILE]], %[[SV1]][%{{.*}}]
    // CHECK: memref.store %[[TILE]], %[[SV0]][%{{.*}}]
    memref.store %tile, %s1[%c0] : memref<2x!ttcore.tile<32x32, f32>, #l1>
    memref.store %tile, %s0[%c0] : memref<2x!ttcore.tile<32x32, f32>, #l1>
  }
  // CHECK-NOT: d2m.scratch_allocate
  return
}

// --- Test 5: Generic without scratch_init is unchanged ---

// CHECK-LABEL: func.func @no_scratch_noop
func.func @no_scratch_noop() {
  %in = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  %out = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  // CHECK-NOT: d2m.scratch_init
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
    %alloc_cb6 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %alloc_cb7 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
  }
  return
}

// --- Test 6: Scratch with bf16 tiles ---

// CHECK-LABEL: func.func @scratch_bf16
func.func @scratch_bf16() {
  %in = memref.alloc() : memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #l1>
  %out = memref.alloc() : memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #l1>
  d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%in : memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #l1>)
  outs(%out : memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #l1>) {
  ^bb0():
    %alloc_cb8 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<2x2x!ttcore.tile<32x32, bf16>, #l1>
    %alloc_cb9 = memref.alloc() {d2m.synchronized_buffer = 2} : memref<2x2x!ttcore.tile<32x32, bf16>, #l1>
    // CHECK: %[[SCRATCH:.*]] = memref.alloc() {d2m.scratch_buffer} : memref<1x4x!ttcore.tile<32x32, bf16>, #l1>
    // CHECK-NOT: d2m.scratch_init
    %scratch = memref.alloc() {d2m.scratch_buffer} : memref<1x4x!ttcore.tile<32x32, bf16>, #l1>
    d2m.scratch_init %scratch : memref<1x4x!ttcore.tile<32x32, bf16>, #l1>
    %c0 = arith.constant 0 : index
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
