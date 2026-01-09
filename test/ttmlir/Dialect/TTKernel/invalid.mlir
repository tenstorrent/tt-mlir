// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Test TTKernelTridNocOpTrait validation - bad TRID (read op).
module {
  %minus_one = arith.constant -1 : i32
  %noc0 = arith.constant 0 : i8
  // expected-error @+1 {{trid must be in [0, 15].}}
  ttkernel.noc_async_read_set_trid(%minus_one, %noc0) : (i32, i8) -> ()
}

// -----

// Test TTKernelTridNocOpTrait validation - bad TRID (write op).
module {
  %bad_trid = arith.constant 16 : i32
  %noc0 = arith.constant 0 : i8
  // expected-error @+1 {{trid must be in [0, 15].}}
  ttkernel.noc_async_write_barrier_with_trid(%bad_trid, %noc0) : (i32, i8) -> ()
}

// -----

// Test TTKernelTridNocOpTrait validation - bad NOC.
module {
  %trid = arith.constant 0 : i32
  %two_i8 = arith.constant 2 : i8
  // expected-error @+1 {{noc must be in [0, 1].}}
  ttkernel.noc_async_read_one_packet_with_state_with_trid(%trid, %trid, %trid, %trid, %two_i8)
      : (i32, i32, i32, i32, i8) -> ()
}

// -----

// Test ResetNocTridBarrierCounterOp validation - bad NOC.
module {
  %mask = arith.constant -1 : i32
  %bad_noc = arith.constant 2 : i8
  // expected-error @+1 {{noc must be in [0, 1].}}
  ttkernel.reset_noc_trid_barrier_counter(%mask, %bad_noc) : (i32, i8) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// TensorAccessorArgsOp parse error tests.
//===----------------------------------------------------------------------===//

// Test: missing opening paren.
func.func @test_tensor_accessor_args_missing_lparen() {
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{expected '('}}
  %args = ttkernel.TensorAccessorArgs %c0, %c0)
  return
}

// -----

// Test: missing '=' after 'prev' keyword.
func.func @test_tensor_accessor_args_missing_prev_equal() {
  %c0 = arith.constant 0 : i32
  %args1 = ttkernel.TensorAccessorArgs(%c0, %c0)
  // expected-error @+1 {{expected '='}}
  %args2 = ttkernel.TensorAccessorArgs(prev %args1)
  return
}

// -----

// Test: missing comma between operands.
func.func @test_tensor_accessor_args_missing_comma() {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  // expected-error @+1 {{expected ','}}
  %args = ttkernel.TensorAccessorArgs(%c0 %c5)
  return
}

// -----

// Test: missing closing paren.
func.func @test_tensor_accessor_args_missing_rparen() {
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{expected ')'}}
  %args = ttkernel.TensorAccessorArgs(%c0, %c0
  return
}

// -----

// Test: missing '=' after cta_expr keyword.
func.func @test_tensor_accessor_args_missing_cta_expr_equal() {
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{expected '='}}
  %args = ttkernel.TensorAccessorArgs(%c0, %c0) cta_expr "foo"
  return
}

// -----

// Test: missing '=' after crta_expr keyword.
func.func @test_tensor_accessor_args_missing_crta_expr_equal() {
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{expected '='}}
  %args = ttkernel.TensorAccessorArgs(%c0, %c0) crta_expr "bar"
  return
}

// -----

// Test: wrong type for prev_args operand (should be !ttkernel.TensorAccessorArgs).
func.func @test_tensor_accessor_args_wrong_prev_type() {
  // expected-note @+1 {{prior use here}}
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{use of value '%c0' expects different type than prior uses: '!ttkernel.TensorAccessorArgs' vs 'i32'}}
  %args = ttkernel.TensorAccessorArgs(prev = %c0)
  return
}

// -----

// Test: wrong type for cta_base operand (should be i32).
func.func @test_tensor_accessor_args_wrong_cta_type() {
  // expected-note @+1 {{prior use here}}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  // expected-error @+1 {{use of value '%c0' expects different type than prior uses: 'i32' vs 'index'}}
  %args = ttkernel.TensorAccessorArgs(%c0, %c1)
  return
}

// -----

// Test: malformed attr-dict.
func.func @test_tensor_accessor_args_malformed_attr_dict() {
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{expected attribute value}}
  %args = ttkernel.TensorAccessorArgs(%c0, %c0) {foo =
  return
}
