// RUN: ttmlir-opt --split-input-file --convert-ttkernel-to-emitc --verify-diagnostics %s

// Tests error cases for TensorAccessorArgs conversion
// When TensorAccessorArgs pattern fails to match (non-constant operands),
// the conversion fails to legalize

// Test: cta_base must be constant when not using prev_args or cta_expr
// expected-error @+1 {{failed to legalize operation 'func.func'}}
func.func @test_non_constant_cta_base(%arg0: i32) -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %crta = arith.constant 0 : i32
  %bank_address = arith.constant 303104 : i32
  %page_size = arith.constant 32 : i32

  // Using non-constant %arg0 for cta_base without prev_args or cta_expr
  // This should fail pattern matching and cause legalization failure
  %args = "ttkernel.TensorAccessorArgs"(%arg0, %crta) : (i32, i32) -> !ttkernel.TensorAccessorArgs
  %tensor_accessor = "ttkernel.TensorAccessor"(%args, %bank_address, %page_size) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor

  return
}

// -----

// Test: crta_base must be constant when not using prev_args or crta_expr
// expected-error @+1 {{failed to legalize operation 'func.func'}}
func.func @test_non_constant_crta_base(%arg0: i32) -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %cta = arith.constant 0 : i32
  %bank_address = arith.constant 303104 : i32
  %page_size = arith.constant 32 : i32

  // Using non-constant %arg0 for crta_base without prev_args or crta_expr
  // This should fail pattern matching and cause legalization failure
  %args = "ttkernel.TensorAccessorArgs"(%cta, %arg0) : (i32, i32) -> !ttkernel.TensorAccessorArgs
  %tensor_accessor = "ttkernel.TensorAccessor"(%args, %bank_address, %page_size) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor

  return
}
