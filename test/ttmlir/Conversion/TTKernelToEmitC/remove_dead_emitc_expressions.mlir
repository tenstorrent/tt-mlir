// RUN: ttmlir-opt --remove-dead-emitc-expressions %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @erase_dead_chain
  // CHECK-NOT: emitc.constant
  // CHECK-NOT: emitc.add
  // CHECK-NOT: emitc.cast
  // CHECK: return
  func.func @erase_dead_chain(%arg0: i32) {
    %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1 = emitc.add %arg0, %0 : (i32, i32) -> i32
    %2 = emitc.cast %1 : i32 to ui32
    return
  }

  // CHECK-LABEL: func.func @erase_dead_expression
  // CHECK-NOT: emitc.expression
  // CHECK: return
  func.func @erase_dead_expression() {
    %0 = emitc.expression  : () -> i32 {
      %1 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
      %2 = emitc.add %1, %1 : (i32, i32) -> i32
      emitc.yield %2 : i32
    }
    return
  }

  // CHECK-LABEL: func.func @erase_expression_after_consumer
  // CHECK-NOT: emitc.expression
  // CHECK-NOT: emitc.constant
  // CHECK-NOT: emitc.add
  // CHECK: return
  func.func @erase_expression_after_consumer(%arg0: i32) {
    %0 = emitc.expression  : () -> i32 {
      %1 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
      emitc.yield %1 : i32
    }
    %2 = emitc.add %arg0, %0 : (i32, i32) -> i32
    return
  }

  // CHECK-LABEL: func.func @erase_dead_variable_assign
  // CHECK-NOT: emitc.variable
  // CHECK-NOT: emitc.assign
  // CHECK-NOT: emitc.add
  // CHECK: return
  func.func @erase_dead_variable_assign(%arg0: i32) {
    %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
    %1 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2 = emitc.add %arg0, %1 : (i32, i32) -> i32
    emitc.assign %2 : i32 to %0 : !emitc.lvalue<i32>
    return
  }

  // CHECK-LABEL: func.func @keep_read_variable
  // CHECK: emitc.variable
  // CHECK: emitc.load
  // CHECK: return
  func.func @keep_read_variable() -> i32 {
    %0 = "emitc.variable"() <{value = 1 : i32}> : () -> !emitc.lvalue<i32>
    %1 = emitc.load %0 : !emitc.lvalue<i32>
    return %1 : i32
  }

  // CHECK-LABEL: func.func @keep_side_effectful_assign_rhs
  // CHECK-NOT: emitc.variable
  // CHECK-NOT: emitc.assign
  // CHECK: emitc.call_opaque "foo"
  // CHECK: return
  func.func @keep_side_effectful_assign_rhs() {
    %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
    %1 = emitc.call_opaque "foo"() : () -> i32
    emitc.assign %1 : i32 to %0 : !emitc.lvalue<i32>
    return
  }

  // CHECK-LABEL: func.func @keep_used
  // CHECK: %[[CST:.*]] = "emitc.constant"
  // CHECK: %[[SUM:.*]] = emitc.add %arg0, %[[CST]]
  // CHECK: return %[[SUM]]
  func.func @keep_used(%arg0: i32) -> i32 {
    %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1 = emitc.add %arg0, %0 : (i32, i32) -> i32
    return %1 : i32
  }

  // CHECK-LABEL: func.func @keep_side_effectful_call
  // CHECK: emitc.call_opaque "foo"
  // CHECK: return
  func.func @keep_side_effectful_call(%arg0: i32) {
    %0 = emitc.call_opaque "foo"(%arg0) : (i32) -> i32
    return
  }

  // CHECK-LABEL: func.func @keep_side_effectful_expression
  // CHECK: emitc.expression
  // CHECK: call_opaque "foo"
  // CHECK: return
  func.func @keep_side_effectful_expression() {
    %0 = emitc.expression  : () -> i32 {
      %1 = emitc.call_opaque "foo"() : () -> i32
      emitc.yield %1 : i32
    }
    return
  }

  // CHECK-LABEL: func.func @keep_opaque_constant
  // CHECK: "emitc.constant"
  // CHECK: return
  func.func @keep_opaque_constant() {
    %0 = "emitc.constant"() <{value = #emitc.opaque<"FOO">}> : () -> !emitc.opaque<"int">
    return
  }
}
