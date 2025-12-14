// RUN: ttmlir-opt --convert-ttkernel-to-emitc --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func @trid_read_path
// CHECK: %[[TRID:.*]] = "emitc.constant"() <{value = 3 : i32}> : () -> i32
// CHECK-NEXT: %[[NOC_IDX:.*]] = "emitc.constant"() <{value = 0 : i8}> : () -> i8
// CHECK-NEXT: %[[X:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK-NEXT: %[[Y:.*]] = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
// CHECK-NEXT: %[[DST_L1:.*]] = "emitc.constant"() <{value = 256 : i32}> : () -> i32
// CHECK-NEXT: %[[SRC_BASE:.*]] = "emitc.constant"() <{value = 1024 : i32}> : () -> i32
// CHECK-NEXT: %[[SRC_ADDR:.*]] = "emitc.constant"() <{value = 64 : i32}> : () -> i32
// CHECK-NEXT: %[[NOC_ADDR:.*]] = emitc.call_opaque "get_noc_addr"(%[[X]], %[[Y]], %[[DST_L1]]) : (!emitc.size_t, !emitc.size_t, i32) -> i64
// CHECK-NEXT: emitc.call_opaque "noc_async_read_set_trid"(%[[TRID]], %[[NOC_IDX]]) : (i32, i8) -> ()
// CHECK-NEXT: emitc.call_opaque "noc_async_read_one_packet_with_state_with_trid"(%[[SRC_BASE]], %[[SRC_ADDR]], %[[DST_L1]], %[[TRID]], %[[NOC_IDX]]) : (i32, i32, i32, i32, i8) -> ()
// CHECK-NEXT: emitc.call_opaque "noc_async_read_barrier_with_trid"(%[[TRID]], %[[NOC_IDX]]) : (i32, i8) -> ()
func.func @trid_read_path() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %trid = arith.constant 3 : i32
  %noc_idx = arith.constant 0 : i8
  %x = arith.constant 0 : index
  %y = arith.constant 1 : index
  %dst_l1 = arith.constant 256 : i32
  %src_base = arith.constant 1024 : i32
  %src_addr = arith.constant 64 : i32
  %noc_addr = "ttkernel.get_noc_addr"(%x, %y, %dst_l1) : (index, index, i32) -> !ttkernel.noc_addr
  "ttkernel.noc_async_read_set_trid"(%trid, %noc_idx) : (i32, i8) -> ()
  "ttkernel.noc_async_read_one_packet_with_state_with_trid"(%src_base, %src_addr, %dst_l1, %trid, %noc_idx) : (i32, i32, i32, i32, i8) -> ()
  "ttkernel.noc_async_read_barrier_with_trid"(%trid, %noc_idx) : (i32, i8) -> ()
  return
}

// -----

// CHECK-LABEL: func @trid_write_path
// CHECK: %[[TRID:.*]] = "emitc.constant"() <{value = 3 : i32}> : () -> i32
// CHECK-NEXT: %[[MASK:.*]] = "emitc.constant"() <{value = -1 : i32}> : () -> i32
// CHECK-NEXT: %[[NOC_IDX:.*]] = "emitc.constant"() <{value = 0 : i8}> : () -> i8
// CHECK-NEXT: %[[X:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK-NEXT: %[[Y:.*]] = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
// CHECK-NEXT: %[[DST:.*]] = "emitc.constant"() <{value = 512 : i32}> : () -> i32
// CHECK-NEXT: %[[SIZE:.*]] = "emitc.constant"() <{value = 128 : i32}> : () -> i32
// CHECK-NEXT: %[[NOC_ADDR:.*]] = emitc.call_opaque "get_noc_addr"(%[[X]], %[[Y]], %[[DST]]) : (!emitc.size_t, !emitc.size_t, i32) -> i64
// CHECK-NEXT: emitc.call_opaque "noc_async_write_set_trid"(%[[TRID]], %[[NOC_IDX]]) : (i32, i8) -> ()
// CHECK-NEXT: emitc.call_opaque "noc_async_write_one_packet_with_trid"(%[[DST]], %[[NOC_ADDR]], %[[SIZE]], %[[TRID]], %[[NOC_IDX]]) : (i32, i64, i32, i32, i8) -> ()
// CHECK-NEXT: emitc.call_opaque "noc_async_write_barrier_with_trid"(%[[TRID]], %[[NOC_IDX]]) : (i32, i8) -> ()
// CHECK-NEXT: emitc.call_opaque "reset_noc_trid_barrier_counter"(%[[MASK]], %[[NOC_IDX]]) : (i32, i8) -> ()
func.func @trid_write_path() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %trid = arith.constant 3 : i32
  %mask = arith.constant -1 : i32
  %noc_idx = arith.constant 0 : i8
  %x = arith.constant 0 : index
  %y = arith.constant 1 : index
  %dst = arith.constant 512 : i32
  %size = arith.constant 128 : i32
  %noc_addr = "ttkernel.get_noc_addr"(%x, %y, %dst) : (index, index, i32) -> !ttkernel.noc_addr
  "ttkernel.noc_async_write_set_trid"(%trid, %noc_idx) : (i32, i8) -> ()
  "ttkernel.noc_async_write_one_packet_with_trid"(%dst, %noc_addr, %size, %trid, %noc_idx) : (i32, !ttkernel.noc_addr, i32, i32, i8) -> ()
  "ttkernel.noc_async_write_barrier_with_trid"(%trid, %noc_idx) : (i32, i8) -> ()
  "ttkernel.reset_noc_trid_barrier_counter"(%mask, %noc_idx) : (i32, i8) -> ()
  return
}
