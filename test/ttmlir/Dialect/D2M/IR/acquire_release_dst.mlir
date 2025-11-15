// RUN: ttmlir-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: @valid_acquire_release_single
#dst_ = #ttcore.memory_space<dst>
func.func @valid_acquire_release_single() {
  // CHECK-NEXT: d2m.acquire_dst
  %dst = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst_>
  // CHECK-NEXT: d2m.release_dst
  d2m.release_dst %dst : memref<4x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

// CHECK-LABEL: @valid_acquire_release_dst
#dst_ = #ttcore.memory_space<dst>
func.func @valid_acquire_release_dst() {
  // CHECK-NEXT: d2m.acquire_dst
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  // CHECK-NEXT: d2m.release_dst
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

// CHECK-LABEL: @valid_multiple_acquire_release
#dst_ = #ttcore.memory_space<dst>
func.func @valid_multiple_acquire_release() {
  // CHECK: d2m.acquire_dst
  %dst0 = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  // CHECK-NEXT: d2m.release_dst
  d2m.release_dst %dst0 : memref<2x!ttcore.tile<32x32, f32>, #dst_>
  // CHECK-NEXT: d2m.acquire_dst
  %dst1 = d2m.acquire_dst() : memref<3x!ttcore.tile<32x32, f32>, #dst_>
  // CHECK-NEXT: d2m.release_dst
  d2m.release_dst %dst1 : memref<3x!ttcore.tile<32x32, f32>, #dst_>
  return
}

