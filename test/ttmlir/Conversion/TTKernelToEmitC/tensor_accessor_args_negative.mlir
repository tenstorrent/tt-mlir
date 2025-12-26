// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Tests error cases for TensorAccessorArgs validation
// The op verifier checks that operands are constants when not using chaining/constexpr

// Test: cta_base must be constant when not using prev_args or cta_expr
func.func @test_non_constant_cta_base(%arg0: i32) -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %crta = arith.constant 0 : i32

  // expected-error @+1 {{cta_base must be a constant when prev_args and cta_expr are not provided}}
  %args = "ttkernel.TensorAccessorArgs"(%arg0, %crta) : (i32, i32) -> !ttkernel.TensorAccessorArgs

  return
}

// -----

// Test: crta_base must be constant when not using prev_args or crta_expr
func.func @test_non_constant_crta_base(%arg0: i32) -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %cta = arith.constant 0 : i32

  // expected-error @+1 {{crta_base must be a constant when prev_args and crta_expr are not provided}}
  %args = "ttkernel.TensorAccessorArgs"(%cta, %arg0) : (i32, i32) -> !ttkernel.TensorAccessorArgs

  return
}
