// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Tests error cases for TensorAccessorArgs validation
// The op verifier checks that operands are constants when not using chaining/constexpr

// Test: prev_args and cta_base/crta_base are mutually exclusive (using generic format)
func.func @test_prev_args_with_bases(%arg0: !ttkernel.TensorAccessorArgs) -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %cta = arith.constant 0 : i32
  %crta = arith.constant 0 : i32

  // expected-error @+1 {{cta_base and crta_base should not be provided when using prev_args}}
  %args = "ttkernel.TensorAccessorArgs"(%cta, %crta, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 1>}> : (i32, i32, !ttkernel.TensorAccessorArgs) -> !ttkernel.TensorAccessorArgs

  return
}

// -----

// Test: both cta_base and crta_base are required when not using prev_args (using generic format)
func.func @test_missing_crta_base() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %cta = arith.constant 0 : i32

  // expected-error @+1 {{both cta_base and crta_base are required when prev_args is not provided}}
  %args = "ttkernel.TensorAccessorArgs"(%cta) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i32) -> !ttkernel.TensorAccessorArgs

  return
}

// -----

// Test: cta_base must be constant when not using cta_expr
func.func @test_non_constant_cta_base(%arg0: i32) -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %crta = arith.constant 0 : i32

  // expected-error @+1 {{cta_base must be a constant when cta_expr is not provided}}
  %args = ttkernel.TensorAccessorArgs(%arg0, %crta)

  return
}

// -----

// Test: crta_base must be constant when not using crta_expr
func.func @test_non_constant_crta_base(%arg0: i32) -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %cta = arith.constant 0 : i32

  // expected-error @+1 {{crta_base must be a constant when crta_expr is not provided}}
  %args = ttkernel.TensorAccessorArgs(%cta, %arg0)

  return
}
