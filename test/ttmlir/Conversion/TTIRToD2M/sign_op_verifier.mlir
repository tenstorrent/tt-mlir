// RUN: ttmlir-opt %s -verify-diagnostics

// Test that d2m.tile_sign verifier rejects invalid input types

module {
  func.func @test_sign_non_tile_error() {
    %0 = arith.constant 0.0 : f16
    // expected-error @+1 {{operand #0 must be TT tile, but got 'f16'}}
    %1 = "d2m.tile_sign"(%0) : (f16) -> (f16)
    return
  }

  func.func @test_sign_f32_error() {
    // expected-error @+1 {{sign operation requires Float16 data type, got f32}}
    %0 = "d2m.tile_sign"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    return
  }

  func.func @test_sign_bf16_error() {
    // expected-error @+1 {{sign operation requires Float16 data type, got bf16}}
    %0 = "d2m.tile_sign"(%0) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    return
  }
}
