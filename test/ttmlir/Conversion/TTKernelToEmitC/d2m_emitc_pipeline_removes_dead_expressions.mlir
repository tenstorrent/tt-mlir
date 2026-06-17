// RUN: ttmlir-opt --d2m-emitc-pipeline %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @pipeline_erases_dead_arith
  // CHECK-NOT: emitc.expression
  // CHECK-NOT: emitc.constant
  // CHECK-NOT: emitc.add
  // CHECK: return
  func.func @pipeline_erases_dead_arith() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 2 : i32
    %2 = arith.addi %0, %1 : i32
    return
  }
}
