// RUN: ttmlir-opt %s | FileCheck %s

#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: @valid_release_dst
func.func @valid_release_dst() {
  // CHECK-NEXT: d2m.acquire_dst
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  // CHECK-NEXT: d2m.release_dst
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}
