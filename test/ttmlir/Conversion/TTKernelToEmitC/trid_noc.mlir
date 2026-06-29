// RUN: ttmlir-opt --convert-ttkernel-to-emitc --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func @trid_read_barrier_static_noc
// CHECK: emitc.verbatim "Noc noc0(0);"
// CHECK-NEXT: %[[TRID:.*]] = "emitc.constant"() <{value = 3 : i32}> : () -> i32
// CHECK-NEXT: %[[NOC_IDX:.*]] = "emitc.constant"() <{value = 0 : i8}> : () -> i8
// CHECK-NEXT: emitc.verbatim "noc0.async_read_barrier<NocOptions::TXN_ID>(NocOptVals{{[{][{]}}.trid = {}{{[}]}});" args %[[TRID]] : i32
func.func @trid_read_barrier_static_noc() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %trid = arith.constant 3 : i32
  %noc_idx = arith.constant 0 : i8
  "ttkernel.noc_async_read_barrier_with_trid"(%trid, %noc_idx) : (i32, i8) -> ()
  return
}

// -----

// CHECK-LABEL: func @trid_write_path
// CHECK: emitc.verbatim "UnicastEndpoint unicast_ep;"
// CHECK-NEXT: emitc.verbatim "Noc noc0(0);"
// CHECK-NEXT: %[[TRID:.*]] = "emitc.constant"() <{value = 3 : i32}> : () -> i32
// CHECK-NEXT: %[[NOC_IDX:.*]] = "emitc.constant"() <{value = 0 : i8}> : () -> i8
// CHECK-NEXT: %[[X:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK-NEXT: %[[Y:.*]] = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
// CHECK-NEXT: %[[DST:.*]] = "emitc.constant"() <{value = 512 : i32}> : () -> i32
// CHECK-NEXT: %[[SIZE:.*]] = "emitc.constant"() <{value = 128 : i32}> : () -> i32
// CHECK-NEXT: emitc.verbatim "noc0.async_write<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(
// CHECK-SAME: NocOptVals{{[{][{]}}.trid = {}{{[}]}}
// CHECK-NEXT: emitc.verbatim "noc0.async_write_barrier<NocOptions::TXN_ID>(NocOptVals{{[{][{]}}.trid = {}{{[}]}});" args %[[TRID]] : i32
func.func @trid_write_path() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %trid = arith.constant 3 : i32
  %noc_idx = arith.constant 0 : i8
  %x = arith.constant 0 : index
  %y = arith.constant 1 : index
  %dst = arith.constant 512 : i32
  %size = arith.constant 128 : i32
  ttkernel.noc_async_write_one_packet_with_trid(%dst, core[%x, %y], %dst, %size, %trid, noc %noc_idx) : (i32, index, index, i32, i32, i32, i8) -> ()
  "ttkernel.noc_async_write_barrier_with_trid"(%trid, %noc_idx) : (i32, i8) -> ()
  return
}

// -----

// A TRID barrier with an explicit NOC operand uses the selected static NoC
// object. The trid is the sole verbatim argument.
// CHECK-LABEL: func @trid_barrier_explicit_noc
// CHECK: emitc.verbatim "Noc noc0(0);"
// CHECK: %[[TRID:.*]] = "emitc.constant"() <{value = 3 : i32}> : () -> i32
// CHECK: %[[NOC_IDX:.*]] = "emitc.constant"() <{value = 0 : i8}> : () -> i8
// CHECK: emitc.verbatim "noc0.async_write_barrier<NocOptions::TXN_ID>(NocOptVals{{[{][{]}}.trid = {}{{[}]}});" args %[[TRID]] : i32
func.func @trid_barrier_explicit_noc() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %trid = arith.constant 3 : i32
  %noc_idx = arith.constant 0 : i8
  "ttkernel.noc_async_write_barrier_with_trid"(%trid, %noc_idx) : (i32, i8) -> ()
  return
}

// -----

// A TRID barrier with a dynamic (non-constant) NOC id emits a temporary
// `Noc({})`. The NOC id fills the `Noc({})` placeholder and the trid fills the
// barrier-call placeholder, so both must appear as verbatim args in that order.
// CHECK-LABEL: func @trid_barrier_dynamic_noc
// CHECK: %[[TRID:.*]] = "emitc.constant"() <{value = 3 : i32}> : () -> i32
// CHECK: %[[NOC:.*]] = emitc.cast
// CHECK: emitc.verbatim "Noc({}).async_read_barrier<NocOptions::TXN_ID>(NocOptVals{{[{][{]}}.trid = {}{{[}]}});" args %[[NOC]], %[[TRID]] : i8, i32
func.func @trid_barrier_dynamic_noc() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %trid = arith.constant 3 : i32
  %noc_arg = arith.constant 262400 : i32
  %noc_ptr = ttkernel.reinterpret_cast(%noc_arg) : (i32) -> (!ttkernel.l1_addr_ptr<8>)
  %noc_offset = arith.constant 0 : i32
  %noc = ttkernel.load_from_l1(%noc_ptr, %noc_offset) : (!ttkernel.l1_addr_ptr<8>, i32) -> i8
  "ttkernel.noc_async_read_barrier_with_trid"(%trid, %noc) : (i32, i8) -> ()
  return
}
