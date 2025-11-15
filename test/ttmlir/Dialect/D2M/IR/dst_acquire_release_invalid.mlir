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
func.func @release_of_block_argument(%dst: memref<1x!ttcore.tile<32x32, f32>, #dst_>) {
  // expected-error@below {{operand must be the result of a d2m.acquire_dst operation}}
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Acquire_dst must have corresponding release_dst.
func.func @single_acquire_no_release() {
  // expected-error@below {{result must be used by a corresponding d2m.release_dst operation}}
  %dst0 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Symmetric acquire_dst and release_dst should pass all verifiers.
func.func @valid_symmetric_usage() {
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Multiple valid acquire/release pairs.
func.func @multiple_valid_pairs() {
  %dst0 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>

  %dst1 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst1 : memref<2x!ttcore.tile<32x32, f32>, #dst_>

  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Acquire_dst with multiple uses before release should pass all verifiers.
func.func @acquire_with_multiple_uses_then_release() {
  %dst = d2m.acquire_dst() : memref<3x!ttcore.tile<32x32, f32>, #dst_>

  %c0 = arith.constant 0 : index
  %0 = affine.load %dst[%c0] : memref<3x!ttcore.tile<32x32, f32>, #dst_>
  %1 = affine.load %dst[%c0] : memref<3x!ttcore.tile<32x32, f32>, #dst_>
  affine.store %0, %dst[1] : memref<3x!ttcore.tile<32x32, f32>, #dst_>

  d2m.release_dst %dst : memref<3x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Partial release - first acquire released but second not.
func.func @acquire_partial_release() {
  %dst0 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>

  // expected-error@below {{result must be used by a corresponding d2m.release_dst operation}}
  %dst1 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  return
}
