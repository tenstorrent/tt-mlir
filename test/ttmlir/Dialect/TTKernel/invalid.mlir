// RUN: ttmlir-opt --verify-diagnostics %s

// Test TTKernelTridNocOpTrait validation - bad TRID (read op)
module {
  %minus_one = arith.constant -1 : i32
  %noc0 = arith.constant 0 : i8
  // expected-error @+1 {{trid must be in [0, 15].}}
  ttkernel.noc_async_read_set_trid(%minus_one, %noc0) : (i32, i8) -> ()
}

// Test TTKernelTridNocOpTrait validation - bad TRID (write op)
module {
  %bad_trid = arith.constant 16 : i32
  %noc0 = arith.constant 0 : i8
  // expected-error @+1 {{trid must be in [0, 15].}}
  ttkernel.noc_async_write_barrier_with_trid(%bad_trid, %noc0) : (i32, i8) -> ()
}

// Test TTKernelTridNocOpTrait validation - bad NOC
module {
  %trid = arith.constant 0 : i32
  %two_i8 = arith.constant 2 : i8
  // expected-error @+1 {{noc must be in [0, 1].}}
  ttkernel.noc_async_read_one_packet_with_state_with_trid(%trid, %trid, %trid, %trid, %two_i8)
      : (i32, i32, i32, i32, i8) -> ()
}

// Test ResetNocTridBarrierCounterOp validation - bad NOC
module {
  %mask = arith.constant -1 : i32
  %bad_noc = arith.constant 2 : i8
  // expected-error @+1 {{noc must be in [0, 1].}}
  ttkernel.reset_noc_trid_barrier_counter(%mask, %bad_noc) : (i32, i8) -> ()
}
