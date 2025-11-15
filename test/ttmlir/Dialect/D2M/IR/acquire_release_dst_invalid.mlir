// RUN: ttmlir-opt %s -split-input-file -verify-diagnostics

#dst_ = #ttcore.memory_space<dst>

// Test that acquire_dst must be paired with release_dst
func.func @invalid_acquire_dst_no_release() {
  // expected-error@+1 {{result must be used by a corresponding d2m.release_dst operation}}
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Test that release_dst requires the operand to be from acquire_dst
func.func @invalid_release_dst_not_acquire(%arg0: memref<1x!ttcore.tile<32x32, f32>, #dst_>) {
  // expected-error@+1 {{operand must be the result of a d2m.acquire_dst operation}}
  d2m.release_dst %arg0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

#dst_ = #ttcore.memory_space<dst>

// Test that DST values cannot be used after release_dst
func.func @invalid_release_dst_used_after() {
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  // expected-error@+1 {{DST value used after release_dst operation}}
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %0 = memref.load %dst[] : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

