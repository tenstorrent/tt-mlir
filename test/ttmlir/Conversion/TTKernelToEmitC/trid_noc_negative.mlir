// RUN: ttmlir-opt --verify-diagnostics %s

// Verify that TRID/NOC validation works through the conversion pipeline
// (Comprehensive validation tests are in test/ttmlir/Dialect/TTKernel/invalid.mlir)

module {
  // Verify trait validation is active - bad TRID
  func.func @bad_trid() {
    %bad_trid = arith.constant 16 : i32
    // expected-error @+1 {{trid must be in [0, 15].}}
    "ttkernel.noc_async_read_set_trid"(%bad_trid) : (i32) -> ()
    return
  }

  // Verify trait validation is active - bad NOC
  func.func @bad_noc() {
    %trid_ok = arith.constant 1 : i32
    %bad_noc = arith.constant 2 : i8
    // expected-error @+1 {{noc must be in [0, 1].}}
    "ttkernel.noc_async_write_set_trid"(%trid_ok, %bad_noc) : (i32, i8) -> ()
    return
  }

  // Verify ResetNocTridBarrierCounterOp validation is active
  func.func @bad_noc_reset() {
    %mask = arith.constant -1 : i32
    %bad_noc = arith.constant 2 : i8
    // expected-error @+1 {{noc must be in [0, 1].}}
    "ttkernel.reset_noc_trid_barrier_counter"(%mask, %bad_noc) : (i32, i8) -> ()
    return
  }
}
