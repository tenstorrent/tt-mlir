// RUN: ttmlir-opt %s -split-input-file -verify-diagnostics

#dst_ = #ttcore.memory_space<dst>
module {
  // Test that DST values cannot be used after release_dst
  func.func @invalid_use_after_release() {
    %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
    // expected-error@below {{'d2m.release_dst' op DST value used after release_dst operation}}
    d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
    %0 = memref.load %dst[] : memref<1x!ttcore.tile<32x32, f32>, #dst_>
    return
  }
}

// -----

#dst_ = #ttcore.memory_space<dst>
module {
  // Test that release_dst requires the operand to be from acquire_dst or a block argument
  func.func @invalid_release_dst_not_acquire() {
    %dst = memref.alloc() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
    // expected-error@below {{'d2m.release_dst' op operand must be the result of a d2m.acquire_dst operation or a block argument}}
    d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
    return
  }
}
