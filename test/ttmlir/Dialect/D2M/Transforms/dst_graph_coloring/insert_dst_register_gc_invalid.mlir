// RUN: ttmlir-opt %s --ttcore-register-device -d2m-insert-dst-register-gc -verify-diagnostics -split-input-file

#dst_ = #ttcore.memory_space<dst>

// Test: Empty function should work fine.
func.func @empty_function() {
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Test: Isolated acquire_dst (no liveness interference).
func.func @isolated_acquire_dst() {
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Test: Multiple independent DST allocations with no interference.
func.func @independent_dst_allocations() {
  %dst0 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %dst1 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst1 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Test: DST with direct release (simple case).
func.func @simple_release() {
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Test: Multiple acquire_dst and release_dst (stress test).
func.func @many_acquire_dst() {
  %dst0 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %dst1 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %dst2 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %dst3 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %dst4 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst1 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst2 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst3 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst4 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>
#l1_ = #ttcore.memory_space<l1>

// DST values with different types should be handled correctly.
// Liveness analysis doesn't care about types, only about value identities.
func.func @different_types_dst() {
  %dst_f32 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %dst_f16 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f16>, #dst_>
  d2m.release_dst %dst_f32 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst_f16 : memref<1x!ttcore.tile<32x32, f16>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Test: Function with conditional branches and releases.
func.func @complex_liveness(%cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dst0 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  %dst1 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>

  // Release both before branching.
  d2m.release_dst %dst0 : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst1 : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>

// Test: CB volume exceeds available DST capacity.
// Default capacity for f16 with fullSyncEn=true is 16 tiles.
// This test uses a 5x5 tile CB (25 tiles), which exceeds the 16-tile limit.
module {
  func.func @cb_exceeds_dst_capacity(%in0: memref<1x1x5x5x!ttcore.tile<32x32, f16>, #ttcore.shard<20480x4096, 1>, #l1_>,
                                      %out: memref<1x1x5x5x!ttcore.tile<32x32, f16>, #ttcore.shard<20480x4096, 1>, #l1_>) {
    // expected-error @below {{CB volume exceeds available DST tiles}}
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0 : memref<1x1x5x5x!ttcore.tile<32x32, f16>, #ttcore.shard<20480x4096, 1>, #l1_>)
      outs(%out : memref<1x1x5x5x!ttcore.tile<32x32, f16>, #ttcore.shard<20480x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<5x5x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<5x5x!ttcore.tile<32x32, f16>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<5x5x!ttcore.tile<32x32, f16>, #l1_>> -> memref<5x5x!ttcore.tile<32x32, f16>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<5x5x!ttcore.tile<32x32, f16>, #l1_>> -> memref<5x5x!ttcore.tile<32x32, f16>, #l1_>

      affine.for %i = 0 to 5 {
        affine.for %j = 0 to 5 {
          %v0 = affine.load %mem0[%i, %j] : memref<5x5x!ttcore.tile<32x32, f16>, #l1_>
          %result = "d2m.tile_abs"(%v0) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          affine.store %result, %mem_out[%i, %j] : memref<5x5x!ttcore.tile<32x32, f16>, #l1_>
        }
      }
    }
    return
  }
}
