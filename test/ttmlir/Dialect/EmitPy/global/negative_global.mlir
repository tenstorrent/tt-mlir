// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for EmitPy global ops.

module {
  // CHECK: error: 'emitpy.global' op requires attribute 'initial_value'
  "emitpy.global"() <{sym_name = "missing_initial_value"}> : () -> ()
}

// -----

module {
  func.func @get_global_nonexistent() -> !emitpy.opaque<"[ttnn.Tensor]"> {
    // CHECK: error: 'emitpy.get_global' op 'nonexistent_global' does not reference a valid emitpy.global
    %0 = emitpy.get_global @nonexistent_global : !emitpy.opaque<"[ttnn.Tensor]">
    return %0 : !emitpy.opaque<"[ttnn.Tensor]">
  }
}

// -----

module {
  func.func @assign_global_nonexistent(%arg0: !emitpy.opaque<"[ttnn.Tensor]">) -> !emitpy.opaque<"[ttnn.Tensor]"> {
    // CHECK: error: 'emitpy.assign_global' op 'nonexistent_global' does not reference a valid emitpy.global
    %0 = "emitpy.assign_global"(%arg0) <{name = @nonexistent_global}> : (!emitpy.opaque<"[ttnn.Tensor]">) -> !emitpy.opaque<"[ttnn.Tensor]">
    return %0 : !emitpy.opaque<"[ttnn.Tensor]">
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
  // CHECK: error: 'emitpy.global' op variable name 'global' is not a valid Python identifier
  emitpy.global @global : #emitpy.opaque<"None">
}

// -----

module {
  // CHECK: error: @ identifier expected to start with letter or '_'
  emitpy.global @123invalid : #emitpy.opaque<"None">
}
