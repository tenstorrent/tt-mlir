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
func.func @release_with_block_argument(%dst: memref<1x!ttcore.tile<32x32, f32>, #dst_>) {
  // expected-error@below {{operand must be the result of a d2m.acquire_dst operation}}
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// Acquire_dst requires corresponding release_dst operation.
func.func @acquire_without_release_variant() {
  // expected-error@below {{result must be used by a corresponding d2m.release_dst operation}}
  %dst0 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// DST value used after release in same block should fail.
func.func @dst_used_after_release_same_block() {
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %c0 = arith.constant 0 : index

  // expected-error@below {{DST value used after release_dst operation}}
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>

  // This use comes after release in the same block - should fail
  %val = affine.load %dst[%c0] : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// DST value used in different block (via scf.if) should fail.
func.func @dst_used_cross_block_scf_if(%cond: i1) {
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %c0 = arith.constant 0 : index

  // expected-error@below {{DST value used in nested region after release_dst operation}}
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>

  // This use is in a different block (inside scf.if) - should fail
  scf.if %cond {
    %val = affine.load %dst[%c0] : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  }
  return
}

// DST value used in different block (via scf.for) should fail.
func.func @dst_used_cross_block_scf_for() {
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // expected-error@below {{DST value used in nested region after release_dst operation}}
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>

  // This use is in a different block (inside scf.for loop body) - should fail
  scf.for %i = %c0 to %c2 step %c1 {
    %val = affine.load %dst[%c0] : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  }
  return
}
