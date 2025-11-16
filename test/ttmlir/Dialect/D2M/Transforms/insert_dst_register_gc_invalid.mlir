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

  // Same DST value passed to two different block argument positions.
  cf.br ^bb1(%dst, %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>, memref<1x!ttcore.tile<32x32, f32>, #dst_>)

^bb1(%arg0: memref<1x!ttcore.tile<32x32, f32>, #dst_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #dst_>):
  d2m.release_dst %arg0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
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

  // All passed to next block - all become live-in.
  cf.br ^bb1(%dst0, %dst1, %dst2, %dst3, %dst4 :
    memref<1x!ttcore.tile<32x32, f32>, #dst_>,
    memref<1x!ttcore.tile<32x32, f32>, #dst_>,
    memref<1x!ttcore.tile<32x32, f32>, #dst_>,
    memref<1x!ttcore.tile<32x32, f32>, #dst_>,
    memref<1x!ttcore.tile<32x32, f32>, #dst_>)

^bb1(%arg0: memref<1x!ttcore.tile<32x32, f32>, #dst_>,
     %arg1: memref<1x!ttcore.tile<32x32, f32>, #dst_>,
     %arg2: memref<1x!ttcore.tile<32x32, f32>, #dst_>,
     %arg3: memref<1x!ttcore.tile<32x32, f32>, #dst_>,
     %arg4: memref<1x!ttcore.tile<32x32, f32>, #dst_>):
  d2m.release_dst %arg0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %arg1 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %arg2 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %arg3 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %arg4 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
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

  // Pass to block - should work fine despite type mismatch.
  cf.br ^bb1(%dst_f32, %dst_f16 : memref<1x!ttcore.tile<32x32, f32>, #dst_>, memref<1x!ttcore.tile<32x32, f16>, #dst_>)

^bb1(%arg0: memref<1x!ttcore.tile<32x32, f32>, #dst_>, %arg1: memref<1x!ttcore.tile<32x32, f16>, #dst_>):
  d2m.release_dst %arg0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %arg1 : memref<1x!ttcore.tile<32x32, f16>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Test: Function with simple release of DST values.
func.func @complex_liveness(%cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dst0 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  %dst1 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>

  // Branch with condition - different blocks have different live sets.
  cf.cond_br %cond, ^bb1(%dst0 : memref<2x!ttcore.tile<32x32, f32>, #dst_>),
                    ^bb2(%dst1 : memref<2x!ttcore.tile<32x32, f32>, #dst_>)

^bb1(%arg0: memref<2x!ttcore.tile<32x32, f32>, #dst_>):
  cf.br ^bb3(%arg0 : memref<2x!ttcore.tile<32x32, f32>, #dst_>)

^bb2(%arg1: memref<2x!ttcore.tile<32x32, f32>, #dst_>):
  cf.br ^bb3(%arg1 : memref<2x!ttcore.tile<32x32, f32>, #dst_>)

^bb3(%arg2: memref<2x!ttcore.tile<32x32, f32>, #dst_>):
  d2m.release_dst %arg2 : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  return
}
