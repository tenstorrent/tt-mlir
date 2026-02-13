// RUN: ttmlir-opt %s -split-input-file -verify-diagnostics

// Test: Expression body must have a terminator
func.func @test_expression_no_terminator(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // expected-error @+1 {{must yield a value at termination}}
  %0 = "emitpy.expression"(%arg0) ({
  ^bb0(%a: !emitpy.opaque<"int">):
    %1 = emitpy.call_opaque "relu"(%a) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
  }) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
  return %0 : !emitpy.opaque<"int">
}

// -----

// Test: Yield must provide a value
// Note: This test triggers compiler heap overflow since yield is defined within expression
func.func @test_expression_yield_no_value(%arg0: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // expected-error @+1 {{yielded value not defined within expression}}
  %0 = "emitpy.expression"(%arg0) ({
  ^bb0(%a: !emitpy.opaque<"float">):
    %1 = emitpy.call_opaque "relu"(%a) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    "emitpy.yield"() : () -> ()
  }) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float">
  return %0 : !emitpy.opaque<"float">
}

// -----

// Test: Yielded value must have a defining op
func.func @test_expression_yield_block_arg(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // expected-error @+1 {{yielded value has no defining op}}
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%a: !emitpy.opaque<"int">):
    emitpy.yield %a : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// Test: Yielded value must be defined within the expression
func.func @test_expression_yield_external_value(%arg0: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  %external = emitpy.call_opaque "relu"(%arg0) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float">
  // expected-error @+1 {{yielded value not defined within expression}}
  %0 = emitpy.expression (%arg0) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%a: !emitpy.opaque<"float">):
    emitpy.yield %external : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// -----

// Test: Yielded type must match expression result type
func.func @test_expression_type_mismatch(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // expected-error @+1 {{requires yielded type to match return type}}
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%a: !emitpy.opaque<"int">):
    %1 = emitpy.literal "42" : index
    emitpy.yield %1 : index
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// Test: Operations must implement PyExpressionInterface
func.func @test_expression_unsupported_op(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // expected-error @+1 {{contains an unsupported operation}}
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%a: !emitpy.opaque<"int">):
    // emitpy.assign doesn't implement PyExpressionInterface
    %1 = emitpy.assign %a : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %1 : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// Test: Operations must have exactly one result
func.func @test_expression_op_multiple_results(%arg0: !emitpy.opaque<"int">) -> index {
  // expected-error @+1 {{requires exactly one result for each operation}}
  %0 = emitpy.expression (%arg0) : (!emitpy.opaque<"int">) -> index {
  ^bb0(%a: !emitpy.opaque<"int">):
    // call_opaque can have multiple results
    %1:2 = emitpy.call_opaque "multi_return"(%a) : (!emitpy.opaque<"int">) -> (index, index)
    emitpy.yield %1#0 : index
  }
  return %0 : index
}

// -----

// Test: All operations must be used (no unused operations)
func.func @test_expression_unused_operation(%arg0: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // expected-error @+1 {{contains an unused operation}}
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%a: !emitpy.opaque<"float">):
    %unused = emitpy.call_opaque "relu"(%a) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    %result = emitpy.call_opaque "gelu"(%a) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    emitpy.yield %result : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// -----

// Test: Operations with side effects must be used exactly once
// Note: call_opaque has side effects by default (hasSideEffects() returns true)
func.func @test_expression_side_effect_used_twice(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // expected-error @+1 {{requires exactly one use for operations with side effects}}
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%a: !emitpy.opaque<"int">):
    %side_effect = emitpy.call_opaque "random_dropout"(%a) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %use1 = emitpy.call_opaque "add"(%side_effect, %side_effect) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %use1 : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// Test: Literal operations (which have no side effects) can be used multiple times
func.func @test_expression_literal_multiple_uses(%arg0: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // This should pass - literal has no side effects
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%a: !emitpy.opaque<"float">):
    %c = emitpy.literal "2.0" : index
    // Using literal twice should be fine since it has no side effects
    %mul1 = emitpy.call_opaque "mul_scalar"(%a, %c) : (!emitpy.opaque<"float">, index) -> !emitpy.opaque<"float">
    %mul2 = emitpy.call_opaque "mul_scalar"(%mul1, %c) : (!emitpy.opaque<"float">, index) -> !emitpy.opaque<"float">
    emitpy.yield %mul2 : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// -----

// Test: Complex case with diamond pattern where side effect op is used multiple times
func.func @test_expression_diamond_side_effect(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // expected-error @+1 {{requires exactly one use for operations with side effects}}
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%a: !emitpy.opaque<"int">):
    // dropout has side effects (randomness)
    %dropout = emitpy.call_opaque "dropout"(%a) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %branch1 = emitpy.call_opaque "relu"(%dropout) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %branch2 = emitpy.call_opaque "gelu"(%dropout) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %merged = emitpy.call_opaque "add"(%branch1, %branch2) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %merged : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// Test: Subscript operations can be used in expressions (they implement PyExpressionInterface)
func.func @test_expression_with_subscript(%arg0: !emitpy.opaque<"list[int]">) -> !emitpy.opaque<"int"> {
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"list[int]">) -> !emitpy.opaque<"int"> {
  ^bb0(%list: !emitpy.opaque<"list[int]">):
    %idx = emitpy.literal "0" : index
    %elem = emitpy.subscript %list[%idx] : (!emitpy.opaque<"list[int]">, index) -> !emitpy.opaque<"int">
    %result = emitpy.call_opaque "process"(%elem) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %result : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}
