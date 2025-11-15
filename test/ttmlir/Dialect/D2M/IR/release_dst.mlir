// RUN: ttmlir-opt %s | FileCheck %s

#dst_ = #ttcore.memory_space<dst>

// Valid symmetrical acquire/release pair.
// CHECK-LABEL: @valid_acquire_release_pair
func.func @valid_acquire_release_pair() {
  // CHECK: d2m.acquire_dst
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  // CHECK: d2m.release_dst
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// Multiple acquire/release pairs.
// CHECK-LABEL: @multiple_acquire_release_pairs
func.func @multiple_acquire_release_pairs() {
  // CHECK: d2m.acquire_dst
  %dst0 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  // CHECK: d2m.release_dst
  d2m.release_dst %dst0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>

  // CHECK: d2m.acquire_dst
  %dst1 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  // CHECK: d2m.release_dst
  d2m.release_dst %dst1 : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// release_dst with multiple uses before it.
// CHECK-LABEL: @acquire_used_multiple_times_then_released
func.func @acquire_used_multiple_times_then_released() {
  // CHECK: d2m.acquire_dst
  %dst = d2m.acquire_dst() : memref<3x!ttcore.tile<32x32, f32>, #dst_>

  // Multiple loads from the DST (valid pattern).
  %c0 = arith.constant 0 : index
  %0 = affine.load %dst[%c0] : memref<3x!ttcore.tile<32x32, f32>, #dst_>
  %1 = affine.load %dst[%c0] : memref<3x!ttcore.tile<32x32, f32>, #dst_>

  // CHECK: d2m.release_dst
  d2m.release_dst %dst : memref<3x!ttcore.tile<32x32, f32>, #dst_>
  return
}
