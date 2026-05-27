// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

func.func @test_bitcast_i16_to_f32(%input: i16) {
  // expected-error @below {{requires source and result types with equal bit widths, got 'i16' and 'f32'}}
  %result = ttkernel.bitcast %input : i16 to f32
  return
}

// -----

func.func @test_bitcast_f32_to_i16(%input: f32) {
  // expected-error @below {{requires source and result types with equal bit widths, got 'f32' and 'i16'}}
  %result = ttkernel.bitcast %input : f32 to i16
  return
}

// -----

func.func @test_bitcast_ui16_to_f32(%input: ui16) {
  // expected-error @below {{requires source and result types with equal bit widths, got 'ui16' and 'f32'}}
  %result = ttkernel.bitcast %input : ui16 to f32
  return
}

// -----

func.func @test_bitcast_f32_to_ui16(%input: f32) {
  // expected-error @below {{requires source and result types with equal bit widths, got 'f32' and 'ui16'}}
  %result = ttkernel.bitcast %input : f32 to ui16
  return
}

// -----

func.func @test_bitcast_i32_to_i32(%input: i32) {
  // expected-error @below {{does not support bitcast from 'i32' to 'i32'}}
  %result = ttkernel.bitcast %input : i32 to i32
  return
}

// -----

func.func @test_bitcast_ui16_to_f16(%input: ui16) {
  // expected-error @below {{does not support bitcast from 'ui16' to 'f16'}}
  %result = ttkernel.bitcast %input : ui16 to f16
  return
}

// -----

func.func @test_bitcast_non_scalar_input(%input: vector<2xi32>) {
  // expected-error @below {{'ttkernel.bitcast' op operand #0 must be}}
  %result = ttkernel.bitcast %input : vector<2xi32> to f32
  return
}

// -----

func.func @test_bitcast_non_scalar_result(%input: i32) {
  // expected-error @below {{'ttkernel.bitcast' op result #0 must be}}
  %result = ttkernel.bitcast %input : i32 to vector<2xf32>
  return
}
