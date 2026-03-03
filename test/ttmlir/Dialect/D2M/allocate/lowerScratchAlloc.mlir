// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-scratch-allocate %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#parallel = #ttcore.iterator_type<parallel>

module {

// --- Test 1: Single scratch_allocate replaced by subview of get_scratch_from_cb result ---

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
  ^bb0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %c0 = arith.constant 0 : index
    // CHECK: %[[SCRATCH:.*]] = d2m.get_scratch_from_cb %{{.*}} : <memref<1x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x8x!ttcore.tile<32x32, f32>, #l1>
    // CHECK: %[[SV:.*]] = memref.subview %[[SCRATCH]][0, 0] [1, 1] [1, 1]
    // CHECK-SAME: memref<1x8x!ttcore.tile<32x32, f32>, #l1> to memref<1x!ttcore.tile<32x32, f32>
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

// --- Test 2: Two scratch_allocates with sequential offsets ---
// slot 0 requests 1 tile  -> subview at offset 0
// slot 1 requests 2 tiles -> subview at offset 1

// CHECK-LABEL: func.func @two_scratch_allocates
func.func @two_scratch_allocates() {
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
  ^bb0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %c0 = arith.constant 0 : index
    // CHECK: %[[SCRATCH:.*]] = d2m.get_scratch_from_cb %{{.*}} : <memref<1x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x8x!ttcore.tile<32x32, f32>, #l1>

    // Slot 0: 1 tile at offset 0.
    // CHECK: %[[SV0:.*]] = memref.subview %[[SCRATCH]][0, 0] [1, 1] [1, 1]
    // CHECK-SAME: to memref<1x!ttcore.tile<32x32, f32>
    %s0 = d2m.scratch_allocate {slot = 0 : i64} : memref<1x!ttcore.tile<32x32, f32>, #l1>

    // Slot 1: 2 tiles at offset 1 (after slot 0's 1 tile).
    // CHECK: %[[SV1:.*]] = memref.subview %[[SCRATCH]][0, 1] [1, 2] [1, 1]
    // CHECK-SAME: to memref<2x!ttcore.tile<32x32, f32>
    %s1 = d2m.scratch_allocate {slot = 1 : i64} : memref<2x!ttcore.tile<32x32, f32>, #l1>

    // Verify stores go to the correct subviews.
    %tile = memref.load %s0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
    memref.store %tile, %s0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1>
    memref.store %tile, %s1[%c0] : memref<2x!ttcore.tile<32x32, f32>, #l1>
  }
  // CHECK-NOT: d2m.scratch_allocate
  return
}

// --- Test 3: Generic without scratch_inputs is unchanged ---

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
  ^bb0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):

  }
  return
}

// --- Test 4: Scratch with bf16 tiles ---

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
  ^bb0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, bf16>, #l1>>, %cb1: !d2m.cb<memref<1x4x!ttcore.tile<32x32, bf16>, #l1>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, bf16>, #l1>>):
    %c0 = arith.constant 0 : index
    // CHECK: %[[SCRATCH:.*]] = d2m.get_scratch_from_cb %{{.*}} : <memref<1x4x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x4x!ttcore.tile<32x32, bf16>, #l1>
    // CHECK: memref.subview %[[SCRATCH]][0, 0] [1, 1] [1, 1]
    // CHECK-SAME: memref<1x4x!ttcore.tile<32x32, bf16>, #l1> to memref<1x!ttcore.tile<32x32, bf16>
    %scratch_slot = d2m.scratch_allocate {slot = 0 : i64} : memref<1x!ttcore.tile<32x32, bf16>, #l1>
    %tile = memref.load %scratch_slot[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1>
    memref.store %tile, %scratch_slot[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1>
  }
  // CHECK-NOT: d2m.scratch_allocate
  return
}

}
