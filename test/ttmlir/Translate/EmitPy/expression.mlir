// RUN: ttmlir-translate -mlir-to-python %s | FileCheck %s

// CHECK-LABEL: def test_simple_expression
func.func @test_simple_expression(%arg0: !emitpy.opaque<"int">, %arg1: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: return mul(inputs, inputs_0)
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%a: !emitpy.opaque<"int">, %b: !emitpy.opaque<"int">):
    %1 = emitpy.call_opaque "mul"(%a, %b) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %1 : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// CHECK-LABEL: def test_nested_expression
func.func @test_nested_expression(%arg0: !emitpy.opaque<"float">, %arg1: !emitpy.opaque<"float">, %arg2: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // CHECK: return relu(add(mul(inputs, inputs_0), inputs_1))
  %0 = emitpy.expression(%arg0, %arg1, %arg2) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%a: !emitpy.opaque<"float">, %b: !emitpy.opaque<"float">, %c: !emitpy.opaque<"float">):
    %1 = emitpy.call_opaque "mul"(%a, %b) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    %2 = emitpy.call_opaque "add"(%1, %c) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    %3 = emitpy.call_opaque "relu"(%2) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    emitpy.yield %3 : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// CHECK-LABEL: def test_expression_with_do_not_inline
func.func @test_expression_with_do_not_inline(%arg0: !emitpy.opaque<"float">, %arg1: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // When do_not_inline is set, we should generate intermediate variables
  // CHECK: var_[[V:[0-9]+]] = add(inputs, inputs_0)
  // CHECK: return
  %0 = emitpy.expression(%arg0, %arg1) {do_not_inline} : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%a: !emitpy.opaque<"float">, %b: !emitpy.opaque<"float">):
    %1 = emitpy.call_opaque "add"(%a, %b) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    emitpy.yield %1 : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// CHECK-LABEL: def test_expression_with_literal
func.func @test_expression_with_literal(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: return mul_scalar(inputs, 2)
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%a: !emitpy.opaque<"int">):
    %c = emitpy.literal "2" : index
    %1 = emitpy.call_opaque "mul_scalar"(%a, %c) : (!emitpy.opaque<"int">, index) -> !emitpy.opaque<"int">
    emitpy.yield %1 : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// CHECK-LABEL: def test_expression_with_subscript
func.func @test_expression_with_subscript(%arg0: !emitpy.opaque<"list">) -> !emitpy.opaque<"int"> {
  // CHECK: return process(inputs[0])
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"list">) -> !emitpy.opaque<"int"> {
  ^bb0(%list: !emitpy.opaque<"list">):
    %idx = emitpy.literal "0" : index
    %elem = emitpy.subscript %list[%idx] : (!emitpy.opaque<"list">, index) -> !emitpy.opaque<"int">
    %result = emitpy.call_opaque "process"(%elem) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %result : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// CHECK-LABEL: def test_expression_no_operands
func.func @test_expression_no_operands() -> !emitpy.opaque<"int"> {
  // CHECK: return get_default_value()
  %0 = emitpy.expression : -> !emitpy.opaque<"int"> {
  ^bb0():
    %1 = emitpy.call_opaque "get_default_value"() : () -> !emitpy.opaque<"int">
    emitpy.yield %1 : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// CHECK-LABEL: def test_expression_chain_of_ops
func.func @test_expression_chain_of_ops(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: return transform4(transform3(transform2(transform1(inputs))))
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

// CHECK-LABEL: def test_expression_multiple_literals_reuse
func.func @test_expression_multiple_literals_reuse(%arg0: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // Literals can be reused multiple times since they have no side effects
  // CHECK: return scale(scale(scale(inputs, 10.5), 10.5), 10.5)
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%val: !emitpy.opaque<"float">):
    %c = emitpy.literal "10.5" : index
    %1 = emitpy.call_opaque "scale"(%val, %c) : (!emitpy.opaque<"float">, index) -> !emitpy.opaque<"float">
    %2 = emitpy.call_opaque "scale"(%1, %c) : (!emitpy.opaque<"float">, index) -> !emitpy.opaque<"float">
    %3 = emitpy.call_opaque "scale"(%2, %c) : (!emitpy.opaque<"float">, index) -> !emitpy.opaque<"float">
    emitpy.yield %3 : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// CHECK-LABEL: def test_expression_tree_structure
func.func @test_expression_tree_structure(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // Tree-like computation structure
  // CHECK: return merge(left_transform(inputs), right_transform(inputs))
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%root: !emitpy.opaque<"int">):
    %left = emitpy.call_opaque "left_transform"(%root) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %right = emitpy.call_opaque "right_transform"(%root) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %merged = emitpy.call_opaque "merge"(%left, %right) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %merged : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// CHECK-LABEL: def test_expression_with_multiple_subscripts
func.func @test_expression_with_multiple_subscripts(%arg0: !emitpy.opaque<"list[int]">, %arg1: !emitpy.opaque<"dict">) -> !emitpy.opaque<"int"> {
  // CHECK: return combine(inputs[0], inputs_0[1])
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"list[int]">, !emitpy.opaque<"dict">) -> !emitpy.opaque<"int"> {
  ^bb0(%list: !emitpy.opaque<"list[int]">, %dict: !emitpy.opaque<"dict">):
    %idx0 = emitpy.literal "0" : index
    %idx1 = emitpy.literal "1" : index
    %elem1 = emitpy.subscript %list[%idx0] : (!emitpy.opaque<"list[int]">, index) -> !emitpy.opaque<"int">
    %elem2 = emitpy.subscript %dict[%idx1] : (!emitpy.opaque<"dict">, index) -> !emitpy.opaque<"int">
    %result = emitpy.call_opaque "combine"(%elem1, %elem2) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %result : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// CHECK-LABEL: def test_expression_complex_dag
func.func @test_expression_complex_dag(%arg0: !emitpy.opaque<"int">, %arg1: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // DAG computation pattern
  // CHECK: return op5(op3(op1(inputs), op2(inputs_0)), op4(inputs_0))
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%x: !emitpy.opaque<"int">, %y: !emitpy.opaque<"int">):
    %a = emitpy.call_opaque "op1"(%x) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %b = emitpy.call_opaque "op2"(%y) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %c = emitpy.call_opaque "op3"(%a, %b) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %d = emitpy.call_opaque "op4"(%y) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %e = emitpy.call_opaque "op5"(%c, %d) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %e : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// CHECK-LABEL: def test_expression_with_boolean
func.func @test_expression_with_boolean(%arg0: !emitpy.opaque<"bool">, %arg1: !emitpy.opaque<"bool">) -> !emitpy.opaque<"bool"> {
  // CHECK: return logical_and(inputs, inputs_0)
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"bool">, !emitpy.opaque<"bool">) -> !emitpy.opaque<"bool"> {
  ^bb0(%a: !emitpy.opaque<"bool">, %b: !emitpy.opaque<"bool">):
    %result = emitpy.call_opaque "logical_and"(%a, %b) : (!emitpy.opaque<"bool">, !emitpy.opaque<"bool">) -> !emitpy.opaque<"bool">
    emitpy.yield %result : !emitpy.opaque<"bool">
  }
  return %0 : !emitpy.opaque<"bool">
}

// CHECK-LABEL: def test_expression_mixed_types
func.func @test_expression_mixed_types(%arg0: !emitpy.opaque<"int">, %arg1: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  // CHECK: return combine(int_to_float(inputs), inputs_0)
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"int">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
  ^bb0(%a: !emitpy.opaque<"int">, %b: !emitpy.opaque<"float">):
    %conv_a = emitpy.call_opaque "int_to_float"(%a) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"float">
    %result = emitpy.call_opaque "combine"(%conv_a, %b) : (!emitpy.opaque<"float">, !emitpy.opaque<"float">) -> !emitpy.opaque<"float">
    emitpy.yield %result : !emitpy.opaque<"float">
  }
  return %0 : !emitpy.opaque<"float">
}

// CHECK-LABEL: def test_expression_deeply_nested
func.func @test_expression_deeply_nested(%arg0: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: return f8(f7(f6(f5(f4(f3(f2(f1(inputs))))))))
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

// CHECK-LABEL: def test_expression_with_list_operations
func.func @test_expression_with_list_operations(%arg0: !emitpy.opaque<"list">) -> !emitpy.opaque<"list"> {
  // CHECK: return sorted(reversed(append(inputs, 42)))
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"list">) -> !emitpy.opaque<"list"> {
  ^bb0(%lst: !emitpy.opaque<"list">):
    %val = emitpy.literal "42" : index
    %appended = emitpy.call_opaque "append"(%lst, %val) : (!emitpy.opaque<"list">, index) -> !emitpy.opaque<"list">
    %reversed = emitpy.call_opaque "reversed"(%appended) : (!emitpy.opaque<"list">) -> !emitpy.opaque<"list">
    %sorted = emitpy.call_opaque "sorted"(%reversed) : (!emitpy.opaque<"list">) -> !emitpy.opaque<"list">
    emitpy.yield %sorted : !emitpy.opaque<"list">
  }
  return %0 : !emitpy.opaque<"list">
}

// CHECK-LABEL: def test_expression_with_dictionary
func.func @test_expression_with_dictionary(%arg0: !emitpy.opaque<"dict">) -> !emitpy.opaque<"int"> {
  // CHECK: return get(inputs, key, default)
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"dict">) -> !emitpy.opaque<"int"> {
  ^bb0(%d: !emitpy.opaque<"dict">):
    %key_lit = emitpy.literal "key" : index
    %default_lit = emitpy.literal "default" : index
    %value = emitpy.call_opaque "get"(%d, %key_lit, %default_lit) : (!emitpy.opaque<"dict">, index, index) -> !emitpy.opaque<"int">
    emitpy.yield %value : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// CHECK-LABEL: def test_multiple_expressions_in_function
func.func @test_multiple_expressions_in_function(%arg0: !emitpy.opaque<"int">, %arg1: !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  // CHECK: return abs(mul(add(inputs, inputs_0), 2))
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%x: !emitpy.opaque<"int">, %y: !emitpy.opaque<"int">):
    %sum = emitpy.call_opaque "add"(%x, %y) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %sum : !emitpy.opaque<"int">
  }

  %1 = emitpy.expression(%0) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%val: !emitpy.opaque<"int">):
    %two = emitpy.literal "2" : index
    %doubled = emitpy.call_opaque "mul"(%val, %two) : (!emitpy.opaque<"int">, index) -> !emitpy.opaque<"int">
    emitpy.yield %doubled : !emitpy.opaque<"int">
  }

  %2 = emitpy.expression(%1) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int"> {
  ^bb0(%num: !emitpy.opaque<"int">):
    %abs_val = emitpy.call_opaque "abs"(%num) : (!emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    emitpy.yield %abs_val : !emitpy.opaque<"int">
  }

  return %2 : !emitpy.opaque<"int">
}

// CHECK-LABEL: def test_expression_with_python_builtins
func.func @test_expression_with_python_builtins(%arg0: !emitpy.str, %arg1: !emitpy.str) -> !emitpy.str {
  // CHECK: return str(max(len(inputs), len(inputs_0)))
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.str, !emitpy.str) -> !emitpy.str {
  ^bb0(%s1: !emitpy.str, %s2: !emitpy.str):
    %len1 = emitpy.call_opaque "len"(%s1) : (!emitpy.str) -> !emitpy.opaque<"int">
    %len2 = emitpy.call_opaque "len"(%s2) : (!emitpy.str) -> !emitpy.opaque<"int">
    %max_len = emitpy.call_opaque "max"(%len1, %len2) : (!emitpy.opaque<"int">, !emitpy.opaque<"int">) -> !emitpy.opaque<"int">
    %str_result = emitpy.call_opaque "str"(%max_len) : (!emitpy.opaque<"int">) -> !emitpy.str
    emitpy.yield %str_result : !emitpy.str
  }
  return %0 : !emitpy.str
}

// CHECK-LABEL: def test_expression_with_tuple_operations
func.func @test_expression_with_tuple_operations(%arg0: !emitpy.opaque<"tuple">) -> !emitpy.opaque<"int"> {
  // CHECK: return sum(map(inputs))
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"tuple">) -> !emitpy.opaque<"int"> {
  ^bb0(%tup: !emitpy.opaque<"tuple">):
    %mapped = emitpy.call_opaque "map"(%tup) : (!emitpy.opaque<"tuple">) -> !emitpy.opaque<"list">
    %summed = emitpy.call_opaque "sum"(%mapped) : (!emitpy.opaque<"list">) -> !emitpy.opaque<"int">
    emitpy.yield %summed : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}

// CHECK-LABEL: def test_expression_with_set_operations
func.func @test_expression_with_set_operations(%arg0: !emitpy.opaque<"set">, %arg1: !emitpy.opaque<"set">) -> !emitpy.opaque<"set"> {
  // CHECK: return intersection(union(inputs, inputs_0), difference(inputs, inputs_0))
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"set">, !emitpy.opaque<"set">) -> !emitpy.opaque<"set"> {
  ^bb0(%set1: !emitpy.opaque<"set">, %set2: !emitpy.opaque<"set">):
    %union_set = emitpy.call_opaque "union"(%set1, %set2) : (!emitpy.opaque<"set">, !emitpy.opaque<"set">) -> !emitpy.opaque<"set">
    %diff_set = emitpy.call_opaque "difference"(%set1, %set2) : (!emitpy.opaque<"set">, !emitpy.opaque<"set">) -> !emitpy.opaque<"set">
    %result = emitpy.call_opaque "intersection"(%union_set, %diff_set) : (!emitpy.opaque<"set">, !emitpy.opaque<"set">) -> !emitpy.opaque<"set">
    emitpy.yield %result : !emitpy.opaque<"set">
  }
  return %0 : !emitpy.opaque<"set">
}

// CHECK-LABEL: def test_expression_with_complex_numbers
func.func @test_expression_with_complex_numbers(%arg0: !emitpy.opaque<"complex">, %arg1: !emitpy.opaque<"complex">) -> !emitpy.opaque<"complex"> {
  // CHECK: return add(mul(inputs, inputs_0), conjugate(inputs_0))
  %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque<"complex">, !emitpy.opaque<"complex">) -> !emitpy.opaque<"complex"> {
  ^bb0(%c1: !emitpy.opaque<"complex">, %c2: !emitpy.opaque<"complex">):
    %prod = emitpy.call_opaque "mul"(%c1, %c2) : (!emitpy.opaque<"complex">, !emitpy.opaque<"complex">) -> !emitpy.opaque<"complex">
    %conj = emitpy.call_opaque "conjugate"(%c2) : (!emitpy.opaque<"complex">) -> !emitpy.opaque<"complex">
    %result = emitpy.call_opaque "add"(%prod, %conj) : (!emitpy.opaque<"complex">, !emitpy.opaque<"complex">) -> !emitpy.opaque<"complex">
    emitpy.yield %result : !emitpy.opaque<"complex">
  }
  return %0 : !emitpy.opaque<"complex">
}

// CHECK-LABEL: def test_expression_with_none_type
func.func @test_expression_with_none_type(%arg0: !emitpy.opaque<"Optional[int]">) -> !emitpy.opaque<"int"> {
  // CHECK: return or_else(inputs, 0)
  %0 = emitpy.expression(%arg0) : (!emitpy.opaque<"Optional[int]">) -> !emitpy.opaque<"int"> {
  ^bb0(%opt: !emitpy.opaque<"Optional[int]">):
    %default = emitpy.literal "0" : index
    %result = emitpy.call_opaque "or_else"(%opt, %default) : (!emitpy.opaque<"Optional[int]">, index) -> !emitpy.opaque<"int">
    emitpy.yield %result : !emitpy.opaque<"int">
  }
  return %0 : !emitpy.opaque<"int">
}
