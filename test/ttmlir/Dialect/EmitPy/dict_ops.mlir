// RUN: ttmlir-opt --split-input-file %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file %s | ttmlir-opt --split-input-file | FileCheck %s
// Test EmitPy dictionary operations: create_dict, set_value_for_dict_key, get_value_for_dict_key

//===----------------------------------------------------------------------===//
// CreateDictOp tests
//===----------------------------------------------------------------------===//

// Test empty dictionary creation using literal_expr
module {
  // CHECK-LABEL: func.func @test_create_empty_dict
  func.func @test_create_empty_dict() -> !emitpy.dict {
    // CHECK: emitpy.create_dict "my_dict" {literal_expr = "{}"} : () -> !emitpy.dict
    %dict = emitpy.create_dict "my_dict" {literal_expr = "{}"} : () -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test untyped dictionary with key-value pairs
module {
  // CHECK-LABEL: func.func @test_create_untyped_dict
  func.func @test_create_untyped_dict(%arg0: index, %arg1: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.dict {
    // CHECK: emitpy.create_dict "cache"(%arg0, %arg1) : (index, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.dict
    %dict = emitpy.create_dict "cache" (%arg0, %arg1) : (index, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test typed dictionary with index keys and tensor values
module {
  // CHECK-LABEL: func.func @test_create_typed_dict_index_tensor
  func.func @test_create_typed_dict_index_tensor(%arg0: index, %arg1: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.dict<index, !emitpy.opaque<"ttnn.Tensor">> {
    // CHECK: emitpy.create_dict "tensor_cache"(%arg0, %arg1) : (index, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.dict<index, !emitpy.opaque<"ttnn.Tensor">>
    %dict = emitpy.create_dict "tensor_cache" (%arg0, %arg1) : (index, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.dict<index, !emitpy.opaque<"ttnn.Tensor">>
    return %dict : !emitpy.dict<index, !emitpy.opaque<"ttnn.Tensor">>
  }
}

// -----

// Test typed dictionary with string keys and tensor values
module {
  // CHECK-LABEL: func.func @test_create_typed_dict_string_tensor
  func.func @test_create_typed_dict_string_tensor(%k: !emitpy.opaque<"str">, %v: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.dict<!emitpy.opaque<"str">, !emitpy.opaque<"ttnn.Tensor">> {
    // CHECK: emitpy.create_dict "tensor_cache_2"(%arg0, %arg1) : (!emitpy.opaque<"str">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.dict<!emitpy.opaque<"str">, !emitpy.opaque<"ttnn.Tensor">>
    %dict = emitpy.create_dict "tensor_cache_2" (%k, %v) : (!emitpy.opaque<"str">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.dict<!emitpy.opaque<"str">, !emitpy.opaque<"ttnn.Tensor">>
    return %dict : !emitpy.dict<!emitpy.opaque<"str">, !emitpy.opaque<"ttnn.Tensor">>
  }
}

// -----

// Test dictionary creation with multiple key-value pairs
module {
  // CHECK-LABEL: func.func @test_create_dict_multiple_items
  func.func @test_create_dict_multiple_items(%k0: index, %v0: !emitpy.opaque<"None">, %k1: index, %v1: !emitpy.opaque<"None">) -> !emitpy.dict {
    // CHECK: emitpy.create_dict "multi_dict"(%arg0, %arg1, %arg2, %arg3) : (index, !emitpy.opaque<"None">, index, !emitpy.opaque<"None">) -> !emitpy.dict
    %dict = emitpy.create_dict "multi_dict" (%k0, %v0, %k1, %v1) : (index, !emitpy.opaque<"None">, index, !emitpy.opaque<"None">) -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test dictionary creation with literal expression
module {
  // CHECK-LABEL: func.func @test_create_dict_literal
  func.func @test_create_dict_literal() -> !emitpy.dict {
    // CHECK: emitpy.create_dict "_CONST_EVAL_CACHE" {literal_expr = "{i: None for i in range(100)}"} : () -> !emitpy.dict
    %dict = emitpy.create_dict "_CONST_EVAL_CACHE" {literal_expr = "{i: None for i in range(100)}"} : () -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

//===----------------------------------------------------------------------===//
// SetValueForDictKeyOp tests
//===----------------------------------------------------------------------===//

// Test setting value with index key
module {
  emitpy.global @_CONST_EVAL_CACHE = #emitpy.opaque<"{}"> : !emitpy.dict

  // CHECK-LABEL: func.func @test_set_value_index_key
  func.func @test_set_value_index_key(%value: !emitpy.opaque<"[ttnn.Tensor]">) {
    %dict = emitpy.global_statement @_CONST_EVAL_CACHE : !emitpy.dict
    // CHECK: emitpy.set_value_for_dict_key "_CONST_EVAL_CACHE" %{{.*}}[5 : index] = %{{.*}} : (!emitpy.dict) -> !emitpy.opaque<"[ttnn.Tensor]">
    emitpy.set_value_for_dict_key "_CONST_EVAL_CACHE" %dict[5 : index] = %value : (!emitpy.dict) -> !emitpy.opaque<"[ttnn.Tensor]">
    return
  }
}

// -----

// Test setting value with string key
module {
  emitpy.global @my_cache = #emitpy.opaque<"{}"> : !emitpy.dict

  // CHECK-LABEL: func.func @test_set_value_string_key
  func.func @test_set_value_string_key(%value: !emitpy.opaque<"ttnn.Tensor">) {
    %dict = emitpy.global_statement @my_cache : !emitpy.dict
    // CHECK: emitpy.set_value_for_dict_key "my_cache" %{{.*}}["tensor_key"] = %{{.*}} : (!emitpy.dict) -> !emitpy.opaque<"ttnn.Tensor">
    emitpy.set_value_for_dict_key "my_cache" %dict["tensor_key"] = %value : (!emitpy.dict) -> !emitpy.opaque<"ttnn.Tensor">
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// GetValueForDictKeyOp tests
//===----------------------------------------------------------------------===//

// Test getting value with index key from pre-populated dict
module {
  emitpy.global @_CONST_EVAL_CACHE = #emitpy.opaque<"{5: None}"> : !emitpy.dict

  // CHECK-LABEL: func.func @test_get_value_index_key
  func.func @test_get_value_index_key() -> !emitpy.opaque<"[ttnn.Tensor]"> {
    %dict = emitpy.global_statement @_CONST_EVAL_CACHE : !emitpy.dict
    // CHECK: emitpy.get_value_for_dict_key "_CONST_EVAL_CACHE" %{{.*}}[5 : index] : (!emitpy.dict) -> !emitpy.opaque<"[ttnn.Tensor]">
    %tensors = emitpy.get_value_for_dict_key "_CONST_EVAL_CACHE" %dict[5 : index] : (!emitpy.dict) -> !emitpy.opaque<"[ttnn.Tensor]">
    return %tensors : !emitpy.opaque<"[ttnn.Tensor]">
  }
}

// -----

// Test getting value with string key from pre-populated dict
module {
  emitpy.global @my_cache = #emitpy.opaque<"{\"tensor_key\": None}"> : !emitpy.dict

  // CHECK-LABEL: func.func @test_get_value_string_key
  func.func @test_get_value_string_key() -> !emitpy.opaque<"ttnn.Tensor"> {
    %dict = emitpy.global_statement @my_cache : !emitpy.dict
    // CHECK: emitpy.get_value_for_dict_key "my_cache" %{{.*}}["tensor_key"] : (!emitpy.dict) -> !emitpy.opaque<"ttnn.Tensor">
    %tensor = emitpy.get_value_for_dict_key "my_cache" %dict["tensor_key"] : (!emitpy.dict) -> !emitpy.opaque<"ttnn.Tensor">
    return %tensor : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

//===----------------------------------------------------------------------===//
// Combined set/get operations test
//===----------------------------------------------------------------------===//

// Test combined set and get operations
module {
  emitpy.global @tensor_cache = #emitpy.opaque<"{}"> : !emitpy.dict

  // CHECK-LABEL: func.func @test_set_then_get
  func.func @test_set_then_get(%input: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    %dict = emitpy.global_statement @tensor_cache : !emitpy.dict
    // CHECK: emitpy.set_value_for_dict_key "tensor_cache" %{{.*}}[42 : index] = %{{.*}}
    emitpy.set_value_for_dict_key "tensor_cache" %dict[42 : index] = %input : (!emitpy.dict) -> !emitpy.opaque<"ttnn.Tensor">
    // CHECK: emitpy.get_value_for_dict_key "tensor_cache" %{{.*}}[42 : index]
    %output = emitpy.get_value_for_dict_key "tensor_cache" %dict[42 : index] : (!emitpy.dict) -> !emitpy.opaque<"ttnn.Tensor">
    return %output : !emitpy.opaque<"ttnn.Tensor">
  }
}
