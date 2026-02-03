// RUN: ttmlir-opt -o %t %s
// RUN: ttmlir-translate --mlir-to-python -o %t2 %t
// RUN: FileCheck %s --input-file=%t2
// Test EmitPy to Python translation for global operations.

module {
  // CHECK: global_var_0 = 5
  emitpy.global @global_var_0 = #emitpy.opaque<"5">
}

// -----

module {
  // CHECK: global_var_1 = None
  emitpy.global @global_var_1 = #emitpy.opaque<"None">

  func.func @set_global_1(%arg0 : !emitpy.opaque<"[ttnn.Tensor]">) -> () {
    // CHECK: def set_global_1([[INS:.*]]):
    // CHECK: global global_var_1
    // CHECK: global_var_1 = [[INS]]
    %0 = emitpy.global_statement @global_var_1 : !emitpy.opaque<"[ttnn.Tensor]">
    emitpy.assign_global @global_var_1 = %arg0 : !emitpy.opaque<"[ttnn.Tensor]">
    return
  }
}

// -----

module {
  // CHECK: global_var_2 = 100
  emitpy.global @global_var_2 = 100 : i64

  func.func @set_global_2() -> () {
    // CHECK: def set_global_2():
    // CHECK: global global_var_2
    // CHECK: global_var_2 = [[VAR:.*]]
    %0 = emitpy.global_statement @global_var_2 : i64
    %1 = "emitpy.constant"() <{value = 110 : i64}> : () -> i64
    emitpy.assign_global @global_var_2 = %1 : i64
    return
  }
}

// -----

module {
  // CHECK: global_var_3 = 0
  emitpy.global @global_var_3 = 0
  // CHECK: global_var_4 = 10
  emitpy.global @global_var_4 = 10

  func.func @multiple_globals() -> () {
    // CHECK: def multiple_globals():
    // CHECK: global global_var_3
    %0 = emitpy.global_statement @global_var_3 : i64
    // CHECK: global global_var_4
    %1 = emitpy.global_statement @global_var_4 : i64
    %2 = emitpy.global_statement @global_var_3 : i64
    %3 = emitpy.global_statement @global_var_4 : i64
    // CHECK: global_var_3 = global_var_4
    emitpy.assign_global @global_var_3 = %3 : i64
    return
  }
}
