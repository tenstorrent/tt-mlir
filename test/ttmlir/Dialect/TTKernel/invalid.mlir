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

// -----

//===----------------------------------------------------------------------===//
// NocSemaphoreIncMulticastOp type-constraint tests.
//===----------------------------------------------------------------------===//

// Test: $addr must be !ttkernel.noc_addr.
func.func @test_noc_semaphore_inc_multicast_wrong_addr_type() {
  // expected-note @+1 {{prior use here}}
  %bad_addr = arith.constant 0 : i32
  %incr = arith.constant 1 : i32
  %num_dests = arith.constant 8 : i32
  // expected-error @+1 {{use of value '%bad_addr' expects different type than prior uses: '!ttkernel.noc_addr' vs 'i32'}}
  "ttkernel.noc_semaphore_inc_multicast"(%bad_addr, %incr, %num_dests) : (!ttkernel.noc_addr, i32, i32) -> ()
  return
}

// -----

// Test: $num_dests must be i32.
func.func @test_noc_semaphore_inc_multicast_wrong_num_dests_type() {
  %x = arith.constant 1 : index
  %y = arith.constant 1 : index
  %off = arith.constant 0 : i32
  %addr = "ttkernel.get_noc_addr"(%x, %y, %off) : (index, index, i32) -> (!ttkernel.noc_addr)
  %incr = arith.constant 1 : i32
  // expected-note @+1 {{prior use here}}
  %bad_num_dests = arith.constant 8 : i64
  // expected-error @+1 {{use of value '%bad_num_dests' expects different type than prior uses: 'i32' vs 'i64'}}
  "ttkernel.noc_semaphore_inc_multicast"(%addr, %incr, %bad_num_dests) : (!ttkernel.noc_addr, i32, i32) -> ()
  return
}

// -----

// Test: optional $noc_id must be i8.
func.func @test_noc_semaphore_inc_multicast_wrong_noc_id_type() {
  %x = arith.constant 1 : index
  %y = arith.constant 1 : index
  %off = arith.constant 0 : i32
  %addr = "ttkernel.get_noc_addr"(%x, %y, %off) : (index, index, i32) -> (!ttkernel.noc_addr)
  %incr = arith.constant 1 : i32
  %num_dests = arith.constant 8 : i32
  // expected-note @+1 {{prior use here}}
  %bad_noc_id = arith.constant 0 : i32
  // expected-error @+1 {{use of value '%bad_noc_id' expects different type than prior uses: 'i8' vs 'i32'}}
  "ttkernel.noc_semaphore_inc_multicast"(%addr, %incr, %num_dests, %bad_noc_id) : (!ttkernel.noc_addr, i32, i32, i8) -> ()
  return
}

// -----

// Test: NocSemaphoreIncOp $addr must be !ttkernel.noc_addr (regression cover for
// the same constraint after adding the optional `posted` attribute).
func.func @test_noc_semaphore_inc_wrong_addr_type() {
  // expected-note @+1 {{prior use here}}
  %bad_addr = arith.constant 0 : i32
  %incr = arith.constant 1 : i32
  // expected-error @+1 {{use of value '%bad_addr' expects different type than prior uses: '!ttkernel.noc_addr' vs 'i32'}}
  "ttkernel.noc_semaphore_inc"(%bad_addr, %incr) <{posted = true}> : (!ttkernel.noc_addr, i32) -> ()
  return
}
