// RUN: ttmlir-opt %s -d2m-insert-dst-register-gc -verify-diagnostics -split-input-file

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
#l1_ = #ttcore.memory_space<l1>

// Test: DST with type mismatch (liveness doesn't care about types) - direct release.
func.func @type_mismatch_dst_with_cf() {
  %dst_f32 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %dst_f16 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f16>, #dst_>

  // Can have different types - both released.
  d2m.release_dst %dst_f32 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst_f16 : memref<1x!ttcore.tile<32x32, f16>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Test: Function with simple release of DST values.
func.func @complex_liveness(%cond: i1) {
  %dst0 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  %dst1 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>

  d2m.release_dst %dst0 : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst1 : memref<2x!ttcore.tile<32x32, f32>, #dst_>

  return
}
