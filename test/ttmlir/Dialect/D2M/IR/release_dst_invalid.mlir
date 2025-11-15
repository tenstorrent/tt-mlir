// RUN: ttmlir-opt %s -verify-diagnostics -split-input-file

#dst_ = #ttcore.memory_space<dst>

// Acquire_dst without release_dst should trigger verifier error.
func.func @acquire_without_release() {
  // expected-error@below {{result must be used by a corresponding d2m.release_dst operation}}
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Release_dst with block argument should fail - operand not from acquire_dst.
// Block arguments are not direct results from acquire_dst operations.
func.func @release_with_block_argument(%dst: memref<1x!ttcore.tile<32x32, f32>, #dst_>) {
  // expected-error@below {{operand must be the result of a d2m.acquire_dst operation}}
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Acquire_dst requires corresponding release_dst operation.
func.func @acquire_without_release_variant() {
  // expected-error@below {{result must be used by a corresponding d2m.release_dst operation}}
  %dst0 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  return
}
