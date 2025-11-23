// RUN: ttmlir-opt %s -verify-diagnostics

#dst_ = #ttcore.memory_space<dst>

// Acquire_dst without release_dst should trigger verifier error.
func.func @acquire_without_release() {
  // expected-error@below {{result must be used by a corresponding d2m.release_dst operation}}
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// Release_dst with block argument should fail - operand not from acquire_dst.
// Block arguments are not direct results from acquire_dst operations.
func.func @release_of_block_argument(%dst: memref<1x!ttcore.tile<32x32, f32>, #dst_>) {
  // expected-error@below {{operand must be the result of a d2m.acquire_dst operation}}
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// Acquire_dst must have corresponding release_dst.
func.func @single_acquire_no_release() {
  // expected-error@below {{result must be used by a corresponding d2m.release_dst operation}}
  %dst0 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// Partial release - first acquire released but second not.
func.func @acquire_partial_release() {
  %dst0 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>

  // expected-error@below {{result must be used by a corresponding d2m.release_dst operation}}
  %dst1 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  return
}
