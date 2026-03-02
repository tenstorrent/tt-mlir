// RUN: ttmlir-opt --split-input-file %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// Test EmitPy if op.

module {
  emitpy.global @_cached = #emitpy.opaque<"None">

  func.func @compute_and_cache(%arg0: !emitpy.opaque<"dict">,
                                %arg1: !emitpy.opaque<"[ttnn.Tensor]">) -> () {
    // CHECK: emitpy.if "not {}" args %{{.*}} : (!emitpy.opaque<"dict">) {
    // CHECK:   emitpy.call_opaque "forward_const_eval_0"
    // CHECK:   emitpy.assign_global @_cached
    // CHECK: }
    emitpy.if "not {}" args %arg0 : (!emitpy.opaque<"dict">) {
      %0 = emitpy.call_opaque "forward_const_eval_0"(%arg1) : (!emitpy.opaque<"[ttnn.Tensor]">) -> !emitpy.opaque<"dict">
      emitpy.assign_global @_cached = %0 : !emitpy.opaque<"dict">
    }
    return
  }
}

// -----

module {
  emitpy.global @_a = #emitpy.opaque<"None">

  func.func @if_multiple_placeholders(%arg0: !emitpy.opaque<"dict">,
                                       %arg1: !emitpy.opaque<"int">,
                                       %arg2: !emitpy.opaque<"dict">) -> () {
    // CHECK: emitpy.if "{} is None and {} == 0" args %{{.*}}, %{{.*}} : (!emitpy.opaque<"dict">, !emitpy.opaque<"int">) {
    // CHECK: }
    emitpy.if "{} is None and {} == 0" args %arg0, %arg1 : (!emitpy.opaque<"dict">, !emitpy.opaque<"int">) {
      emitpy.assign_global @_a = %arg2 : !emitpy.opaque<"dict">
    }
    return
  }
}

// -----

module {
  emitpy.global @_outer = #emitpy.opaque<"None">
  emitpy.global @_inner = #emitpy.opaque<"None">

  func.func @nested_if(%arg0: !emitpy.opaque<"dict">,
                        %arg1: !emitpy.opaque<"[ttnn.Tensor]">,
                        %arg2: !emitpy.opaque<"dict">,
                        %arg3: !emitpy.opaque<"[ttnn.Tensor]">) -> () {
    // CHECK: emitpy.if "{} is None" args %{{.*}} : (!emitpy.opaque<"dict">) {
    // CHECK:   emitpy.if "{} is None" args %{{.*}} : (!emitpy.opaque<"[ttnn.Tensor]">) {
    // CHECK:     emitpy.assign_global @_inner
    // CHECK:   }
    // CHECK:   emitpy.assign_global @_outer
    // CHECK: }
    emitpy.if "{} is None" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.if "{} is None" args %arg1 : (!emitpy.opaque<"[ttnn.Tensor]">) {
        emitpy.assign_global @_inner = %arg3 : !emitpy.opaque<"[ttnn.Tensor]">
      }
      emitpy.assign_global @_outer = %arg2 : !emitpy.opaque<"dict">
    }
    return
  }
}

// -----

module {
  emitpy.global @_cached = #emitpy.opaque<"None">

  func.func @if_else(%arg0: !emitpy.opaque<"dict">,
                      %arg1: !emitpy.opaque<"dict">) -> () {
    // CHECK: emitpy.if "{} is None" args %{{.*}} : (!emitpy.opaque<"dict">) {
    // CHECK:   emitpy.assign_global @_cached
    // CHECK: } else {
    // CHECK:   emitpy.call_opaque "use_cached"
    // CHECK: }
    emitpy.if "{} is None" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.assign_global @_cached = %arg1 : !emitpy.opaque<"dict">
    } else {
      emitpy.call_opaque "use_cached"(%arg0) : (!emitpy.opaque<"dict">) -> ()
    }
    return
  }
}

// -----

module {
  emitpy.global @_cached = #emitpy.opaque<"None">

  func.func @elif_chain(%arg0: !emitpy.opaque<"dict">,
                         %arg1: !emitpy.opaque<"int">,
                         %arg2: !emitpy.opaque<"dict">) -> () {
    // CHECK: emitpy.if "{} is None" args %{{.*}} : (!emitpy.opaque<"dict">) {
    // CHECK:   emitpy.assign_global @_cached
    // CHECK: } else {
    // CHECK:   emitpy.if "{} > 0" args %{{.*}} : (!emitpy.opaque<"int">) {
    // CHECK:     emitpy.call_opaque "refresh"
    // CHECK:   }
    // CHECK: }
    emitpy.if "{} is None" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.assign_global @_cached = %arg2 : !emitpy.opaque<"dict">
    } else {
      emitpy.if "{} > 0" args %arg1 : (!emitpy.opaque<"int">) {
        emitpy.call_opaque "refresh"(%arg2) : (!emitpy.opaque<"dict">) -> ()
      }
    }
    return
  }
}

// -----

module {
  emitpy.global @_cached = #emitpy.opaque<"None">

  func.func @elif_with_else(%arg0: !emitpy.opaque<"dict">,
                              %arg1: !emitpy.opaque<"int">,
                              %arg2: !emitpy.opaque<"dict">) -> () {
    // CHECK: emitpy.if "{} is None" args %{{.*}} : (!emitpy.opaque<"dict">) {
    // CHECK:   emitpy.assign_global @_cached
    // CHECK: } else {
    // CHECK:   emitpy.if "{} > 0" args %{{.*}} : (!emitpy.opaque<"int">) {
    // CHECK:     emitpy.call_opaque "refresh"
    // CHECK:   } else {
    // CHECK:     emitpy.call_opaque "fallback"
    // CHECK:   }
    // CHECK: }
    emitpy.if "{} is None" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.assign_global @_cached = %arg2 : !emitpy.opaque<"dict">
    } else {
      emitpy.if "{} > 0" args %arg1 : (!emitpy.opaque<"int">) {
        emitpy.call_opaque "refresh"(%arg2) : (!emitpy.opaque<"dict">) -> ()
      } else {
        emitpy.call_opaque "fallback"() : () -> ()
      }
    }
    return
  }
}

// -----

module {
  func.func @if_no_args() -> () {
    // CHECK: emitpy.if "True" {
    // CHECK:   emitpy.call_opaque "do_something"
    // CHECK: }
    emitpy.if "True" {
      emitpy.call_opaque "do_something"() : () -> ()
    }
    return
  }
}

// -----

module {
  func.func @if_no_args_with_else() -> () {
    // CHECK: emitpy.if "True" {
    // CHECK:   emitpy.call_opaque "branch_a"
    // CHECK: } else {
    // CHECK:   emitpy.call_opaque "branch_b"
    // CHECK: }
    emitpy.if "True" {
      emitpy.call_opaque "branch_a"() : () -> ()
    } else {
      emitpy.call_opaque "branch_b"() : () -> ()
    }
    return
  }
}
