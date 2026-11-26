// RUN: ttmlir-opt %s --form-deduplicated-emitc-expressions | FileCheck %s
// RUN: ttmlir-opt %s --form-deduplicated-emitc-expressions --mlir-print-ir-after-all -o %t 2> %t.dump
// RUN: ttmlir-opt %t.dump -o %t.reparsed

module {
  // CHECK-LABEL: func.func @deduplicate_expression_operands
  // CHECK: %[[RESULT:.*]] = emitc.expression %[[X:.*]], %[[Y:.*]], %[[ADDR:.*]], %[[NOC:.*]] : (!emitc.size_t, !emitc.size_t, i32, i8) -> i64 {
  // CHECK-NEXT: %[[MCAST:.*]] = call_opaque "get_noc_multicast_addr"(%[[X]], %[[Y]], %[[X]], %[[ADDR]], %[[NOC]]) : (!emitc.size_t, !emitc.size_t, !emitc.size_t, i32, i8) -> i64
  // CHECK-NEXT: yield %[[MCAST]] : i64
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[RESULT]] : i64
  func.func @deduplicate_expression_operands(
      %x: !emitc.size_t, %y: !emitc.size_t, %addr: i32, %noc: i8) -> i64 {
    %mcast = emitc.call_opaque "get_noc_multicast_addr"(%x, %y, %x, %addr, %noc) : (!emitc.size_t, !emitc.size_t, !emitc.size_t, i32, i8) -> i64
    return %mcast : i64
  }
}
