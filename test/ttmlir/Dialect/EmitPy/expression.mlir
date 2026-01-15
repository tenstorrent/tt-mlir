// RUN: ttmlir-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_simple_expression
func.func @test_simple_expression(%arg0: !emitpy.opaque<"int">, %arg1: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%a: !emitpy.opaque<"int">, %b: !emitpy.opaque<"int">):
    %1 = emitpy.call_opaque "mul"(%a, %b) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %1 : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// CHECK-LABEL: func.func @test_nested_expression
func.func @test_nested_expression(%arg0: !emitpy.opaque<"float">, %arg1: !emitpy.opaque<"float">, %arg2: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0, %arg1, %arg2) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%a: !emitpy.opaque<"float">, %b: !emitpy.opaque<"float">, %c: !emitpy.opaque<"float">):
    %1 = emitpy.call_opaque "mul"(%a, %b) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    %2 = emitpy.call_opaque "add"(%1, %c) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    %3 = emitpy.call_opaque "relu"(%2) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    emitpy.yield %3 : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// -----

// CHECK-LABEL: func.func @test_expression_with_do_not_inline
func.func @test_expression_with_do_not_inline(%arg0: !emitpy.opaque<"float">, %arg1: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0, %arg1) {do_not_inline} : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%a: !emitpy.opaque<"float">, %b: !emitpy.opaque<"float">):
    %1 = emitpy.call_opaque "add"(%a, %b) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    emitpy.yield %1 : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// -----

// CHECK-LABEL: func.func @test_expression_with_literal
func.func @test_expression_with_literal(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%a: !emitpy.opaque<"int">):
    %c = emitpy.literal "2" : index
    %1 = emitpy.call_opaque "mul_scalar"(%a, %c) : (!emitpy.opaque<"int">, index) -> !emitpy.opaque<"int">
    emitpy.yield %1 : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// CHECK-LABEL: func.func @test_expression_with_subscript
func.func @test_expression_with_subscript(%arg0: !emitpy.opaque<"list[int]">) -> !emitpy.opaque<"int"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"list[int]">) -> !emitpy.opaque<"int"> {
  ^bb0(%list: !emitpy.opaque<"list[int]">):
    %idx = emitpy.literal "0" : index
    %elem = emitpy.subscript %list[%idx] : (!emitpy.opaque<"list[int]">, index) -> !emitpy.opaque<"int">
    %result = emitpy.call_opaque "increment"(%elem) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %result : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// CHECK-LABEL: func.func @test_expression_no_operands
func.func @test_expression_no_operands() -> !emitpy.opaque<"int"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression : -> !emitpy.opaque<"int"> {
  ^bb0():
    %1 = emitpy.call_opaque "get_default_value"() : () -> !emitpy.opaque<"int">
    emitpy.yield %1 : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// CHECK-LABEL: func.func @test_expression_chain_of_ops
func.func @test_expression_chain_of_ops(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%data: !emitpy.opaque<"int">):
    %1 = emitpy.call_opaque "transform1"(%data) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %2 = emitpy.call_opaque "transform2"(%1) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %3 = emitpy.call_opaque "transform3"(%2) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %4 = emitpy.call_opaque "transform4"(%3) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %4 : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// CHECK-LABEL: func.func @test_expression_with_index_type
func.func @test_expression_with_index_type(%arg0: index, %arg1: index) -> index {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0, %arg1) : (index, index) -> index {
  ^bb0(%a: index, %b: index):
    %c1 = emitpy.literal "1" : index
    %sum = emitpy.call_opaque "add"(%a, %b) : (index, index) -> index
    %result = emitpy.call_opaque "add"(%sum, %c1) : (index, index) -> index
    emitpy.yield %result : index
  }
  return %0 : index
}

// -----

// CHECK-LABEL: func.func @test_expression_multiple_literals_reuse
func.func @test_expression_multiple_literals_reuse(%arg0: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // CHECK: emitpy.expression
  // This tests that literals (which have no side effects) can be used multiple times
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%val: !emitpy.opaque<"float">):
    %c = emitpy.literal "10.5" : index
    %1 = emitpy.call_opaque "scale"(%val, %c) : (!emitpy.opaque<"float">, index) -> !emitpy.opaque<"float">
    %2 = emitpy.call_opaque "scale"(%1, %c) : (!emitpy.opaque<"float">, index) -> !emitpy.opaque<"float">
    // Reusing literal %c again
    %3 = emitpy.call_opaque "scale"(%2, %c) : (!emitpy.opaque<"float">, index) -> !emitpy.opaque<"float">
    emitpy.yield %3 : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// -----

// CHECK-LABEL: func.func @test_expression_tree_structure
func.func @test_expression_tree_structure(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: emitpy.expression
  // This tests a tree-like computation structure (not a simple chain)
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%root: !emitpy.opaque<"int">):
    // Create two branches
    %left = emitpy.call_opaque "left_transform"(%root) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %right = emitpy.call_opaque "right_transform"(%root) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    // Merge branches
    %merged = emitpy.call_opaque "merge"(%left, %right) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %merged : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// CHECK-LABEL: func.func @test_expression_with_multiple_subscripts
func.func @test_expression_with_multiple_subscripts(%arg0: !emitpy.opaque<"list[float]">, %arg1: !emitpy.opaque<"list[float]">) -> !emitpy.opaque<"float"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"list[float]">, !emitpy.opaque<"list[float]">) -> !emitpy.opaque<"float"> {
  ^bb0(%arr1: !emitpy.opaque<"list[float]">, %arr2: !emitpy.opaque<"list[float]">):
    %idx0 = emitpy.literal "0" : index
    %idx1 = emitpy.literal "1" : index
    %elem1 = emitpy.subscript %arr1[%idx0] : (!emitpy.opaque<"list[float]">, index) -> !emitpy.opaque<"float">
    %elem2 = emitpy.subscript %arr2[%idx1] : (!emitpy.opaque<"list[float]">, index) -> !emitpy.opaque<"float">
    %result = emitpy.call_opaque "combine"(%elem1, %elem2) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    emitpy.yield %result : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// -----

// CHECK-LABEL: func.func @test_expression_complex_dag
func.func @test_expression_complex_dag(%arg0: !emitpy.opaque<"int">, %arg1: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: emitpy.expression
  // This creates a DAG (directed acyclic graph) computation pattern
  // Note: call_opaque has side effects, so we can't use values multiple times
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%x: !emitpy.opaque<"int">, %y: !emitpy.opaque<"int">):
    // Multiple paths that reconverge
    %a = emitpy.call_opaque "op1"(%x) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %b = emitpy.call_opaque "op2"(%y) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %c = emitpy.call_opaque "op3"(%a, %b) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %d = emitpy.call_opaque "op4"(%y) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %e = emitpy.call_opaque "op5"(%c, %d) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %e : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// CHECK-LABEL: func.func @test_expression_with_boolean
func.func @test_expression_with_boolean(%arg0: !emitpy.opaque<"bool">, %arg1: !emitpy.opaque<"bool">) -> !emitpy.opaque<"bool"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"bool">, !emitpy.opaque<"bool">) -> !emitpy.opaque<"bool"> {
  ^bb0(%a: !emitpy.opaque<"bool">, %b: !emitpy.opaque<"bool">):
    %result = emitpy.call_opaque "logical_and"(%a, %b) : (!emitpy.opaque<"bool">, !emitpy.opaque<"bool">) -> !emitpy.opaque<"bool">
    emitpy.yield %result : !emitpy.opaque<"bool">
  }
  return %0 : !emitpy.opaque<"bool">
}

// -----

// CHECK-LABEL: func.func @test_expression_mixed_types
func.func @test_expression_mixed_types(%arg0: !emitpy.opaque<"int">, %arg1: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"int">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%a: !emitpy.opaque<"int">, %b: !emitpy.opaque<"float">):
    %conv_a = emitpy.call_opaque "int_to_float"(%a) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"float">
    %result = emitpy.call_opaque "combine"(%conv_a, %b) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    emitpy.yield %result : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// -----

// CHECK-LABEL: func.func @test_expression_deeply_nested
func.func @test_expression_deeply_nested(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: emitpy.expression
  // Test a deeply nested chain of operations
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%x: !emitpy.opaque<"int">):
    %1 = emitpy.call_opaque "f1"(%x) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %2 = emitpy.call_opaque "f2"(%1) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %3 = emitpy.call_opaque "f3"(%2) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %4 = emitpy.call_opaque "f4"(%3) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %5 = emitpy.call_opaque "f5"(%4) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %6 = emitpy.call_opaque "f6"(%5) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %7 = emitpy.call_opaque "f7"(%6) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %8 = emitpy.call_opaque "f8"(%7) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %8 : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// -----

// CHECK-LABEL: func.func @test_expression_with_opaque_return
func.func @test_expression_with_opaque_return(%arg0: !emitpy.opaque<"int">, %arg1: !emitpy.opaque<"int">) -> !emitpy.opaque<"PyObject"> {
  // CHECK: emitpy.expression
  // Test that we can still return opaque types when needed (e.g., Python objects)
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"PyObject"> {
  ^bb0(%a: !emitpy.opaque<"int">, %b: !emitpy.opaque<"int">):
    %result = emitpy.call_opaque "create_tuple"(%a, %b) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"PyObject">
    emitpy.yield %result : !emitpy.opaque<"PyObject">
  }
  return %0 : !emitpy.opaque<"PyObject">
}

// -----

// CHECK-LABEL: func.func @test_expression_with_complex_float
func.func @test_expression_with_complex_float(%arg0: !emitpy.opaque<"complex">, %arg1: !emitpy.opaque<"complex">) -> !emitpy.opaque<"complex"> {
  // CHECK: emitpy.expression
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"complex">, !emitpy.opaque<"complex">) -> !emitpy.opaque<"complex"> {
  ^bb0(%a: !emitpy.opaque<"complex">, %b: !emitpy.opaque<"complex">):
    %result = emitpy.call_opaque "complex_multiply"(%a, %b) : (!emitpy.opaque<"complex">, !emitpy.opaque<"complex">) -> !emitpy.opaque<"complex">
    emitpy.yield %result : !emitpy.opaque<"complex">
  }
  return %0 : !emitpy.opaque<"complex">
}
