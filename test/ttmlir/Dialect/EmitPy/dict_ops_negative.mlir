// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for EmitPy dictionary operations.

//===----------------------------------------------------------------------===//
// CreateDictOp negative tests
//===----------------------------------------------------------------------===//

// Test dictionary name cannot be empty
module {
  func.func @test_dict_empty_name() -> !emitpy.dict {
    // CHECK: error: 'emitpy.create_dict' op variable name must not be empty
    %dict = emitpy.create_dict "" : () -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test dictionary name cannot be a Python keyword
module {
  func.func @test_dict_name_keyword() -> !emitpy.dict {
    // CHECK: error: 'emitpy.create_dict' op variable name must not be a keyword
    %dict = emitpy.create_dict "global" : () -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test dictionary name must start with letter or underscore
module {
  func.func @test_dict_name_invalid_start() -> !emitpy.dict {
    // CHECK: error: 'emitpy.create_dict' op variable name must start with a letter or '_'
    %dict = emitpy.create_dict "123invalid" : () -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test dictionary name cannot contain special characters
module {
  func.func @test_dict_name_special_chars() -> !emitpy.dict {
    // CHECK: error: 'emitpy.create_dict' op variable name may only contain alphanumeric characters and '_'
    %dict = emitpy.create_dict "my$dict" : () -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test cannot have both literal_expr and items
module {
  func.func @test_dict_both_literal_and_items(%k: index, %v: !emitpy.opaque<"None">) -> !emitpy.dict {
    // CHECK: error: 'emitpy.create_dict' op cannot have both literal_expr and items operands
    %dict = emitpy.create_dict "bad_dict" (%k, %v) {literal_expr = "{}"} : (index, !emitpy.opaque<"None">) -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test items must be even count (key-value pairs)
module {
  func.func @test_dict_odd_items(%k: index) -> !emitpy.dict {
    // CHECK: error: 'emitpy.create_dict' op items must be alternating key-value pairs (even count required)
    %dict = emitpy.create_dict "odd_dict" (%k) : (index) -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test literal_expr must not be empty string
module {
  func.func @test_dict_empty_literal() -> !emitpy.dict {
    // CHECK: error: 'emitpy.create_dict' op literal_expr must not be empty
    %dict = emitpy.create_dict "empty_literal" {literal_expr = ""} : () -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test cannot have empty items and empty literal_expr
module {
  func.func @test_dict_empty_items_and_literal() -> !emitpy.dict {
    // CHECK: error: 'emitpy.create_dict' op cannot have both literal_expr and items empty
    %dict = emitpy.create_dict "empty_dict" : () -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

// Test dictionary keys must be index or string type (not dict)
module {
  func.func @test_dict_invalid_key_type(%key: !emitpy.dict, %value: !emitpy.opaque<"int">) -> !emitpy.dict {
    // CHECK: error: 'emitpy.create_dict' op dictionary keys must be index or string type, but got '!emitpy.dict' at position 0
    %dict = emitpy.create_dict "bad_key_dict" (%key, %value) : (!emitpy.dict, !emitpy.opaque<"int">) -> !emitpy.dict
    return %dict : !emitpy.dict
  }
}

// -----

//===----------------------------------------------------------------------===//
// SubscriptOp negative tests
//===----------------------------------------------------------------------===//

// Test cannot use string index on non-dict type
module {
  func.func @test_subscript_string_on_non_dict(%arr: !emitpy.opaque<"[int]">, %key: !emitpy.str) -> !emitpy.opaque<"int"> {
    // CHECK: error: 'emitpy.subscript' op cannot use string index on non-dict type '!emitpy.opaque<"[int]">'
    %elem = emitpy.subscript %arr[%key] : (!emitpy.opaque<"[int]">, !emitpy.str) -> !emitpy.opaque<"int">
    return %elem : !emitpy.opaque<"int">
  }
}

// -----

//===----------------------------------------------------------------------===//
// AssignOp negative tests
//===----------------------------------------------------------------------===//

// Test cannot use string index on non-dict type (subscript assignment)
module {
  func.func @test_assign_string_on_non_dict(%arr: !emitpy.opaque<"[int]">, %key: !emitpy.str, %value: !emitpy.opaque<"int">) {
    // CHECK: error: 'emitpy.assign' op cannot use string index on non-dict type '!emitpy.opaque<"[int]">'
    emitpy.assign %arr[%key] = %value : (!emitpy.opaque<"[int]">, !emitpy.str, !emitpy.opaque<"int">)
    return
  }
}
