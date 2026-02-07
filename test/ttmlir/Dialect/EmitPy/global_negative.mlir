// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for EmitPy global ops.

module {
  // CHECK: error: custom op 'emitpy.global' expected initial value for global variable
  emitpy.global @global_var_missing_initial_value =
}

// -----

module {
  // CHECK: error: custom op 'emitpy.global' expected '=' after symbol name
  emitpy.global @global_var
}

// -----

module {
  // CHECK: error: 'emitpy.global' op variable name must not be a keyword
  emitpy.global @global = #emitpy.opaque<"None">
}

// -----

module {
  // CHECK: error: @ identifier expected to start with letter or '_'
  emitpy.global @123invalid = #emitpy.opaque<"None">
}

// -----

module {
  // CHECK: error: 'emitpy.global' op variable name may only contain alphanumeric characters and '_'
  emitpy.global @in$valid = #emitpy.opaque<"None">
}

// -----

module {
  emitpy.global @typed_global = 0

  func.func @assign_global_type_mismatch(%arg0: !emitpy.opaque<"[ttnn.Tensor]">) -> () {
    // CHECK: error: 'emitpy.assign_global' op value type ('!emitpy.opaque<"[ttnn.Tensor]">') does not match global's type ('i64')
    emitpy.assign_global @typed_global = %arg0 : !emitpy.opaque<"[ttnn.Tensor]">
    return
  }
}

// -----

module {
  emitpy.global @typed_global = 0

  func.func @global_statement_type_mismatch() -> !emitpy.opaque<"[ttnn.Tensor]"> {
    // CHECK: error: 'emitpy.global_statement' op result type ('!emitpy.opaque<"[ttnn.Tensor]">') does not match global's type ('i64')
    %0 = emitpy.global_statement @typed_global : !emitpy.opaque<"[ttnn.Tensor]">
    return %0 : !emitpy.opaque<"[ttnn.Tensor]">
  }
}

// -----

module {
  func.func @assign_global_nonexistent(%arg0: !emitpy.opaque<"[ttnn.Tensor]">) -> () {
    // CHECK: error: 'emitpy.assign_global' op 'nonexistent_global' does not reference a valid emitpy.global
    emitpy.assign_global @nonexistent_global = %arg0 : !emitpy.opaque<"[ttnn.Tensor]">
    return
  }
}

// -----

module {
  emitpy.global @global_var = #emitpy.opaque<"None">

  func.func @assign_another_global_nonexistent(%arg0: !emitpy.opaque<"[ttnn.Tensor]">) -> () {
    // CHECK: error: 'emitpy.assign_global' op 'global_var_1' does not reference a valid emitpy.global
    emitpy.assign_global @global_var_1 = %arg0 : !emitpy.opaque<"[ttnn.Tensor]">
    return
  }
}

// -----

module {
  func.func @global_statement_nonexistent() -> !emitpy.opaque<"[ttnn.Tensor]"> {
    // CHECK: error: 'emitpy.global_statement' op 'nonexistent_global' does not reference a valid emitpy.global
    %0 = emitpy.global_statement @nonexistent_global : !emitpy.opaque<"[ttnn.Tensor]">
    return %0 : !emitpy.opaque<"[ttnn.Tensor]">
  }
}

// -----

module {
  emitpy.global @global_var = #emitpy.opaque<"None">

  func.func @another_global_statement_nonexistent() -> !emitpy.opaque<"[ttnn.Tensor]"> {
    // CHECK: error: 'emitpy.global_statement' op 'global_var_1' does not reference a valid emitpy.global
    %0 = emitpy.global_statement @global_var_1 : !emitpy.opaque<"[ttnn.Tensor]">
    return %0 : !emitpy.opaque<"[ttnn.Tensor]">
  }
}
