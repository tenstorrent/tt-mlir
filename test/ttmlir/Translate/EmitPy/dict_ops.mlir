// RUN: ttmlir-opt -o %t %s
// RUN: ttmlir-translate --mlir-to-python -o %t2 %t
// RUN: FileCheck %s --input-file=%t2
// Test EmitPy to Python translation for dictionary operations:
// create_dict, subscript, set_item

//===----------------------------------------------------------------------===//
// CreateDictOp translation tests
//===----------------------------------------------------------------------===//

module {
  // CHECK-LABEL: def test_empty_dict
  func.func @test_empty_dict() -> !emitpy.dict {
    // CHECK: my_dict = {}
    %dict = emitpy.create_dict "my_dict" {literal_expr = "{}"} : () -> !emitpy.dict
    // CHECK: return my_dict
    return %dict : !emitpy.dict
  }

  // CHECK-LABEL: def test_dict_literal_expr
  func.func @test_dict_literal_expr() -> !emitpy.dict<index, !emitpy.opaque<"None">> {
    // CHECK: _CONST_EVAL_CACHE = {i: None for i in range(100)}
    %dict = emitpy.create_dict "_CONST_EVAL_CACHE" {literal_expr = "{i: None for i in range(100)}"} : () -> !emitpy.dict<index, !emitpy.opaque<"None">>
    // CHECK: return _CONST_EVAL_CACHE
    return %dict : !emitpy.dict<index, !emitpy.opaque<"None">>
  }

  // CHECK-LABEL: def test_dict_with_items
  func.func @test_dict_with_items() -> !emitpy.dict<index, !emitpy.opaque<"None">> {
    %k0 = emitpy.literal "0" : index
    %v0 = "emitpy.constant"() <{value = #emitpy.opaque<"None">}> : () -> !emitpy.opaque<"None">
    // CHECK: cache = {0: {{.*}}}
    %dict = emitpy.create_dict "cache" (%k0, %v0) : (index, !emitpy.opaque<"None">) -> !emitpy.dict<index, !emitpy.opaque<"None">>
    // CHECK: return cache
    return %dict : !emitpy.dict<index, !emitpy.opaque<"None">>
  }

  // CHECK-LABEL: def test_dict_multiple_items
  func.func @test_dict_multiple_items() -> !emitpy.dict<index, !emitpy.opaque<"None">> {
    %k0 = emitpy.literal "0" : index
    %v0 = "emitpy.constant"() <{value = #emitpy.opaque<"None">}> : () -> !emitpy.opaque<"None">
    %k1 = emitpy.literal "1" : index
    %v1 = "emitpy.constant"() <{value = #emitpy.opaque<"None">}> : () -> !emitpy.opaque<"None">
    // CHECK: multi_cache = {0: {{.*}}, 1: {{.*}}}
    %dict = emitpy.create_dict "multi_cache" (%k0, %v0, %k1, %v1) : (index, !emitpy.opaque<"None">, index, !emitpy.opaque<"None">) -> !emitpy.dict<index, !emitpy.opaque<"None">>
    // CHECK: return multi_cache
    return %dict : !emitpy.dict<index, !emitpy.opaque<"None">>
  }

  // CHECK-LABEL: def test_dict_complex_literal
  func.func @test_dict_complex_literal() -> !emitpy.dict {
    // CHECK: config = {"key": "value", "num": 42}
    %dict = emitpy.create_dict "config" {literal_expr = "{\"key\": \"value\", \"num\": 42}"} : () -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

//===----------------------------------------------------------------------===//
// SubscriptOp translation tests
//===----------------------------------------------------------------------===//

module {
  emitpy.global @_CONST_EVAL_CACHE = #emitpy.opaque<"{5: None}"> : !emitpy.dict

  // CHECK-LABEL: def test_get_value_index_key
  func.func @test_get_value_index_key() -> !emitpy.opaque<"[ttnn.Tensor]"> {
    // CHECK: global _CONST_EVAL_CACHE
    %dict = emitpy.global_statement @_CONST_EVAL_CACHE : !emitpy.dict
    %key = emitpy.literal "5" : index
    // CHECK: {{.*}} = _CONST_EVAL_CACHE[5]
    %tensors = emitpy.subscript %dict[%key] : (!emitpy.dict, index) -> !emitpy.opaque<"[ttnn.Tensor]">
    // CHECK: return {{.*}}
    return %tensors : !emitpy.opaque<"[ttnn.Tensor]">
  }
}

// -----

//===----------------------------------------------------------------------===//
// SetItemOp translation tests
//===----------------------------------------------------------------------===//

module {
  emitpy.global @_CONST_EVAL_CACHE = #emitpy.opaque<"{}"> : !emitpy.dict

  // CHECK-LABEL: def test_set_value_index_key
  func.func @test_set_value_index_key(%arg0: !emitpy.opaque<"[ttnn.Tensor]">) {
    // CHECK: global _CONST_EVAL_CACHE
    %dict = emitpy.global_statement @_CONST_EVAL_CACHE : !emitpy.dict
    %key = emitpy.literal "5" : index
    // CHECK: _CONST_EVAL_CACHE[5] = {{.*}}
    emitpy.set_item %dict[%key] = %arg0 : (!emitpy.dict, index, !emitpy.opaque<"[ttnn.Tensor]">)
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Combined set/get operations translation test
//===----------------------------------------------------------------------===//

module {
  emitpy.global @tensor_cache = #emitpy.opaque<"{}"> : !emitpy.dict

  // CHECK-LABEL: def test_set_then_get
  func.func @test_set_then_get(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: global tensor_cache
    %dict = emitpy.global_statement @tensor_cache : !emitpy.dict
    %key = emitpy.literal "42" : index
    // CHECK: tensor_cache[42] = {{.*}}
    emitpy.set_item %dict[%key] = %arg0 : (!emitpy.dict, index, !emitpy.opaque<"ttnn.Tensor">)
    // CHECK: {{.*}} = tensor_cache[42]
    %output = emitpy.subscript %dict[%key] : (!emitpy.dict, index) -> !emitpy.opaque<"ttnn.Tensor">
    // CHECK: return {{.*}}
    return %output : !emitpy.opaque<"ttnn.Tensor">
  }
}
