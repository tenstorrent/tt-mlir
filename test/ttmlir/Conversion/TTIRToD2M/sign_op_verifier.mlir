// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s

// Test that d2m.tile_sign verifier rejects non-f16 types

module {
  // CHECK: error: 'd2m.tile_sign' op sign operation requires Float16 data type, got f32
  func.func @test_sign_f32_error() {
    %0 = "d2m.tile_sign"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    return
  }

  // CHECK: error: 'd2m.tile_sign' op sign operation requires Float16 data type, got bf16
  func.func @test_sign_bf16_error() {
    %0 = "d2m.tile_sign"(%0) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    return
  }
}
