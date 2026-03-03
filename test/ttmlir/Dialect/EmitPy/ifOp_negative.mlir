// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for EmitPy if op.

module {
  func.func @empty_condition(%arg0: !emitpy.opaque<"dict">) -> () {
    // CHECK: error: 'emitpy.if' op condition string must not be empty
    emitpy.if "" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.call_opaque "dummy"() : () -> ()
    }
    return
  }
}

// -----

module {
  emitpy.global @g = #emitpy.opaque<"None">

  func.func @placeholder_count_mismatch(%arg0: !emitpy.opaque<"dict">,
                                         %arg1: !emitpy.opaque<"dict">) -> () {
    // CHECK: error: 'emitpy.if' op requires operands for each placeholder in the condition string
    emitpy.if "{} is None and {}" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.assign_global @g = %arg1 : !emitpy.opaque<"dict">
    }
    return
  }
}

// -----

module {
  func.func @unescaped_brace(%arg0: !emitpy.opaque<"dict">) -> () {
    // CHECK: error: 'emitpy.if' op expected '}' after unescaped '{'
    emitpy.if "{x" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.call_opaque "dummy"() : () -> ()
    }
    return
  }
}

// -----

module {
  func.func @too_many_args_no_placeholder(%arg0: !emitpy.opaque<"dict">) -> () {
    // CHECK: error: 'emitpy.if' op requires operands for each placeholder in the condition string
    emitpy.if "True" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.call_opaque "dummy"() : () -> ()
    }
    return
  }
}
